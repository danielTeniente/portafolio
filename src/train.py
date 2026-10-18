"""Reproducible daylight-only, day-ahead Quito forecasting benchmark."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import importlib.metadata
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
try:
    from .recurrent import RecurrentRegressor
except ImportError:
    from recurrent import RecurrentRegressor

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
NEURAL = ('RNN', 'GRU', 'LSTM')


def load_data(path):
    frame = pd.read_csv(path, parse_dates=['Date_time'])
    if frame.isna().any().any():
        raise ValueError('Missing values found. Ask the data owner how to handle them; no automatic imputation.')
    if frame.Date_time.duplicated().any():
        raise ValueError('Duplicate timestamps found; resolve with the data owner.')
    frame = frame.sort_values('Date_time').set_index('Date_time')
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq='h')
    if not frame.index.equals(expected):
        raise ValueError('Missing hourly timestamps found. Ask the data owner before filling or dropping.')
    if len(frame) % 24 or frame.index[0].hour != 0 or frame.index[-1].hour != 23:
        raise ValueError('Incomplete calendar days; ask the data owner how to handle them.')
    if not np.isfinite(frame.RS).all() or (frame.RS < 0).any():
        raise ValueError('Invalid radiation observations; review with the data owner.')
    return frame


def daily_pairs(frame):
    days = frame.RS.to_numpy(dtype=np.float64).reshape(-1, 24)
    return days[:-1], days[1:], frame.index[24::24]


def make_model(name, param):
    if name in NEURAL:
        return RecurrentRegressor(name, param)
    if name == 'Direct AR':
        return make_pipeline(StandardScaler(), Ridge(alpha=param))
    if name == 'XGBoost':
        return XGBRegressor(n_estimators=120, max_depth=param, learning_rate=.04,
                            tree_method='hist', multi_strategy='one_output_per_tree',
                            subsample=.85, colsample_bytree=.85, reg_lambda=10,
                            n_jobs=2, random_state=SEED)
    cls = RandomForestRegressor if name == 'Random Forest' else ExtraTreesRegressor
    return cls(n_estimators=200, min_samples_leaf=param, max_features=.8,
               n_jobs=2, random_state=SEED)


def scores(actual, pred):
    error = pred - actual
    daily = np.abs(error).mean(axis=1)
    return dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error**2).mean())),
                bias=float(error.mean()), p95_daily_mae=float(np.quantile(daily, .95)),
                daily_total_mae=float(np.abs(error.sum(axis=1)).mean()))


def run(path):
    start = time.perf_counter()
    frame = load_data(path)
    full_x, full_y, dates = daily_pairs(frame)
    n = len(dates)
    boundaries = [int(n * f) for f in (.40, .50, .60, .70, .80)]
    a, c = boundaries[0], boundaries[-1]
    folds = list(zip(boundaries[:-1], boundaries[1:]))
    # Only the earliest training block determines which clock hours are retained.
    # A noise floor of 5 RS units excludes small nighttime sensor readings.
    hours = np.flatnonzero(np.concatenate([full_x[:1], full_y[:a]]).max(axis=0) > 5)
    x, y = full_x[:, hours], full_y[:, hours]
    candidates = {'Direct AR': [1., 100.], 'Random Forest': [3, 10],
                  'Extra Trees': [3, 10], 'XGBoost': [2, 4],
                  'RNN': [8, 16], 'GRU': [8, 16], 'LSTM': [8, 16]}
    rows, tuning, predictions = [], [], {}
    sensitivity, architectures = [], {}
    for name, params in candidates.items():
        begun = time.perf_counter()
        options = []
        for param in params:
            absolute_errors = []
            for fold, (end, stop) in enumerate(folds, 1):
                model = make_model(name, param).fit(x[:end], y[:end])
                pred = np.maximum(0, model.predict(x[end:stop]))
                mae = float(np.abs(pred - y[end:stop]).mean())
                tuning.append(dict(model=name, parameter=param, fold=fold, train_days=end,
                                   validation_days=stop-end, first_date=str(dates[end].date()),
                                   last_date=str(dates[stop-1].date()), mae=mae,
                                   epochs=getattr(model, 'epochs_', None),
                                   search_epochs=getattr(model, 'search_epochs_', None)))
                absolute_errors.extend(np.abs(pred-y[end:stop]).ravel())
            options.append((float(np.mean(absolute_errors)), param))
        validation_mae, chosen = min(options)
        fit_start = time.perf_counter()
        model = make_model(name, chosen).fit(x[:c], y[:c])
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        pred = np.maximum(0, model.predict(x[c:]))
        predict_ms_per_day = 1000*(time.perf_counter()-predict_start)/len(pred)
        predictions[name] = pred
        fold_maes = [r['mae'] for r in tuning if r['model']==name and r['parameter']==chosen]
        rows.append(dict(model=name, parameter=chosen, validation_mae=validation_mae,
                         fold_maes=fold_maes, cv_std=float(np.std(fold_maes)),
                         fit_seconds=fit_seconds, total_seconds=time.perf_counter()-begun,
                         predict_ms_per_day=predict_ms_per_day,
                         epochs=getattr(model, 'epochs_', None),
                         **scores(y[c:], pred)))
        print(f'{name}: test daylight MAE {rows[-1]["mae"]:.2f}', flush=True)
        architectures[name] = architecture(name, chosen, model, len(hours))
        if name in NEURAL:
            sensitivity.append(dict(model=name, seed=SEED, epochs=model.epochs_,
                                    mae=rows[-1]['mae'], fit_seconds=fit_seconds))
            for seed in (7, 123):
                seed_start = time.perf_counter()
                alternative = RecurrentRegressor(name, chosen, seed=seed).fit(x[:c], y[:c], epochs=model.epochs_)
                alternative_seconds = time.perf_counter()-seed_start
                seed_pred = np.maximum(0, alternative.predict(x[c:]))
                sensitivity.append(dict(model=name, seed=seed, epochs=model.epochs_,
                                        mae=scores(y[c:], seed_pred)['mae'], fit_seconds=alternative_seconds))
    predictions['Repeat yesterday'] = x[c:]
    baseline = scores(y[c:], x[c:])
    baseline_folds = [float(np.abs(x[end:stop]-y[end:stop]).mean()) for end,stop in folds]
    rows.append(dict(model='Repeat yesterday', parameter=None, validation_mae=None,
                     fold_maes=baseline_folds, cv_std=float(np.std(baseline_folds)),
                     fit_seconds=0., total_seconds=0., **baseline))
    winner = min(rows[:-1], key=lambda r: r['validation_mae'])['model']
    architectures['Repeat yesterday'] = dict(structure='13 daylight readings → copy → same 13 clock hours next day',
                                             settings='No fitting, scaling, learned weights, or tuning.')
    for row in rows:
        row['skill_pct'] = 100 * (1 - row['mae']/baseline['mae'])
    records = []
    for i, date in enumerate(dates[c:]):
        for j, hour in enumerate(hours):
            records.append(dict(timestamp=str(date + pd.Timedelta(hours=int(hour))),
                                actual=y[c+i, j], **{k: v[i, j] for k, v in predictions.items()}))
    results, docs = ROOT/'results', ROOT/'docs'
    results.mkdir(exist_ok=True); docs.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(results/'metrics.csv', index=False)
    pd.DataFrame(tuning).to_csv(results/'validation.csv', index=False)
    pd.DataFrame(records).to_csv(results/'predictions.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(results/'seed_sensitivity.csv', index=False)
    payload = dict(winner=winner, hours=hours.tolist(), metrics=rows, architectures=architectures,
                   research=json.loads((ROOT/'research.json').read_text(encoding='utf-8')),
                   seed_sensitivity=sensitivity,
                   dates=[str(d.date()) for d in dates[c:]], actual=y[c:].round(3).tolist(),
                   predictions={k: v.round(3).tolist() for k, v in predictions.items()},
                   horizon_mae={k: np.abs(v-y[c:]).mean(axis=0).tolist() for k,v in predictions.items()},
                   audit=dict(rows=len(frame), first=str(frame.index[0]), last=str(frame.index[-1]),
                              missing=0, duplicate_timestamps=0, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              units='RS units (measurement units unverified)', noise_floor=5,
                              data_source=json.loads((ROOT/'data_source.json').read_text(encoding='utf-8')),
                              train_days=c, test_days=n-c, first_test=str(dates[c].date()),
                              folds=[dict(train_days=end, validation_days=stop-end,
                                          first_date=str(dates[end].date()), last_date=str(dates[stop-1].date()))
                                     for end,stop in folds], python=platform.python_version(),
                              packages={p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','matplotlib','torch']},
                              elapsed_seconds=time.perf_counter()-start))
    (results/'report.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (docs/'data.js').write_text('window.REPORT = '+json.dumps(payload)+';', encoding='utf-8')
    plot(payload)
    print('Selected on validation:', winner, flush=True)


def plot(report):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.facecolor':'#f4f6f1','axes.facecolor':'#f4f6f1',
                         'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    ranked = sorted(report['metrics'], key=lambda r:r['mae'], reverse=True)
    axes[0].barh([r['model'] for r in ranked], [r['mae'] for r in ranked],
                 color=['#e2aa39' if r['model']==report['winner'] else '#386b67' for r in ranked])
    axes[0].set(title='Daylight forecast error · lower is better', xlabel='MAE · source RS units')
    for name, values in report['horizon_mae'].items():
        axes[1].plot(report['hours'], values, label=name, linewidth=2)
    axes[1].set(title='Where the forecast is hardest', xlabel='Hour in source timestamps', ylabel='MAE · source RS units')
    axes[1].legend(fontsize=8)
    fig.suptitle('QUITO / Next-day solar radiation', fontsize=18, fontweight='bold')
    fig.savefig(ROOT/'docs'/'benchmark.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    for row in report['metrics']:
        if row['model'] == 'Repeat yesterday':
            continue
        neural = row['model'] in NEURAL
        ax.scatter(row['total_seconds'], row['mae'], s=100,
                   color='#d9a230' if neural else '#26756c', marker='D' if neural else 'o')
        offset = {'Extra Trees':(-55,-18), 'Random Forest':(-85,12), 'XGBoost':(10,10),
                  'GRU':(10,-12)}.get(row['model'], (10,8))
        ax.annotate(row['model'], (row['total_seconds'], row['mae']),
                    xytext=offset, textcoords='offset points', fontsize=10)
    ax.set_xscale('log')
    ax.set(xlabel='Search + final fit seconds · logarithmic scale · CPU, single run',
           ylabel='Test daylight MAE · source RS units',
           title='Does complexity earn its place? Lower and further left is better.')
    ax.margins(x=.3, y=.2)
    ax.grid(alpha=.15)
    fig.text(.5, -.02, 'Gold diamonds: recurrent networks. Green circles: classical models. Seed checks and imports excluded.',
             ha='center', fontsize=9)
    fig.savefig(ROOT/'docs'/'complexity.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def architecture(name, param, model, outputs):
    if name in NEURAL:
        return dict(structure=f'Batch × {outputs} daylight steps × 1 RS feature → {name}({param} hidden units, 1 layer) → last hidden state → Linear({param}, {outputs}) → next-day daylight profile',
                    settings=f'{model.parameter_count_} trainable parameters. Unidirectional; hidden state resets per day; no dropout. '+
                    f'Adam lr=0.01, weight_decay=0.001, gradient norm clip=1, full-batch MSE, CPU / 2 threads. '+
                    f'Final refit: {model.epochs_} epochs; inner search ran {model.search_epochs_} epochs (cap 100, patience 12, MAE min_delta 0.01 RS). '+
                    'Last 20% of each training block selects epochs, then weights and scalar input/target standardization are reset and refitted on the full training block. '+
                    'Only widths 8 and 16 are compared; primary seed 42. Output is inverse-scaled and clipped at zero.')
    if name == 'Direct AR':
        return dict(structure=f'{outputs} daylight lag features → per-feature StandardScaler → Ridge → {outputs} direct outputs',
                    settings=f'Alpha={param}; {outputs*outputs} coefficients + {outputs} intercepts. Squared-error regression with L2 penalty; no recurrent state. Scaler fitted inside each training fold. Negative forecasts clipped at zero.')
    if name == 'XGBoost':
        return dict(structure=f'{outputs} lag features → boosted scalar regression trees for each of {outputs} output hours',
                    settings=f'120 boosting rounds; depth={param}; learning_rate=0.04; hist; one_output_per_tree; subsample=0.85; colsample_bytree=0.85; reg_lambda=10; objective=reg:squarederror; 2 threads; seed 42. No scaling; negative forecasts clipped at zero.')
    return dict(structure=f'{outputs} lag features → 200 multi-output regression trees → average {outputs}-value leaf predictions',
                settings=f'min_samples_leaf={param}; max_features=0.8; unlimited depth; squared-error criterion; 2 threads; seed 42. '+
                ('Random Forest uses bootstrap samples and optimized splits. ' if name=='Random Forest' else 'Extra Trees uses all samples (bootstrap=False) and randomized split thresholds. ')+
                'No scaling; negative forecasts clipped at zero.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'quito_2017_2018 (1).csv')
    run(parser.parse_args().data)
