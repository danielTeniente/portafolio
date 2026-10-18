"""Predeclared synthetic benchmark: no injected events or outcome-selected seeds."""
from pathlib import Path
import json
import time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 7, 123)
ALPHA = 1.5
TRAIN, TEST = 600, 150
FOLDS = ((240,330),(330,420),(420,510),(510,600))
CANDIDATES = {'Direct AR':(1,100), 'Random Forest':(3,10),
              'Extra Trees':(3,10), 'XGBoost':(2,4)}


def generate(seed):
    """Positive periodic signal times independent, mean-one innovations."""
    t = np.arange((TRAIN+TEST+1)*24)
    signal = 1 + .45*np.sin(2*np.pi*t/24) + .15*np.sin(2*np.pi*t/(24*7))
    streams = np.random.SeedSequence(seed).spawn(2)
    light = np.random.default_rng(streams[0]).gamma(4, .25, len(t))
    heavy = (np.random.default_rng(streams[1]).pareto(ALPHA, len(t))+1)*(ALPHA-1)/ALPHA
    light_y, heavy_y = signal*light, signal*heavy
    shifted = light_y.copy()
    change = (TRAIN+1+TEST//2)*24
    shifted[change:] = 3*heavy_y[change:]
    return {'light':light_y.reshape(-1,24), 'pareto':heavy_y.reshape(-1,24),
            'shift':shifted.reshape(-1,24)}


def estimator(name, param):
    if name == 'Direct AR':
        return make_pipeline(StandardScaler(), Ridge(alpha=param))
    if name == 'XGBoost':
        return XGBRegressor(n_estimators=100,max_depth=param,learning_rate=.04,
                            tree_method='hist',subsample=.85,colsample_bytree=.85,
                            reg_lambda=10,n_jobs=2,random_state=42)
    cls = RandomForestRegressor if name=='Random Forest' else ExtraTreesRegressor
    return cls(n_estimators=100,min_samples_leaf=param,max_features=.8,n_jobs=2,random_state=42)


def metrics(y,p):
    error=p-y
    threshold=np.quantile(y,.99)
    return dict(mae=float(np.abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),
                worst_underforecast=float(np.maximum(-error,0).max()),
                top_one_percent_underforecast=float(np.maximum(-error[y>=threshold],0).mean()))


def fit_models(days):
    x,y=days[:-1],days[1:]
    fitted, info = {}, []
    for name,params in CANDIDATES.items():
        begun=time.perf_counter()
        options=[]
        for param in params:
            scores=[]
            for end,stop in FOLDS:
                model=estimator(name,param).fit(x[:end],y[:end])
                scores.append(float(np.abs(np.maximum(0,model.predict(x[end:stop]))-y[end:stop]).mean()))
            options.append((float(np.mean(scores)),param,scores))
        score,param,fold_scores=min(options)
        fitted[name]=estimator(name,param).fit(x[:TRAIN],y[:TRAIN])
        info.append(dict(model=name,parameter=param,validation_mae=score,fold_maes=fold_scores,
                         train_seconds=time.perf_counter()-begun))
    return fitted,info


def run():
    begun=time.perf_counter()
    report=dict(protocol=dict(seeds=list(SEEDS),train_days=TRAIN,test_days=TEST,
                alpha=ALPHA,regime_multiplier=3,change_test_day=TEST//2+1,
                input_hours=24,output_hours=24,folds=FOLDS,
                seed_policy='All three predeclared seeds retained; none selected by outcome.',
                scope='Synthetic operational load, not solar data or published research results.',
                generator='signal(t) = 1 + 0.45 sin(2πt/24) + 0.15 sin(2πt/168); load = signal × innovation.',
                light='Gamma(shape=4, scale=0.25), mean 1, variance 0.25.',
                pareto='Type I Pareto with survival exponent α=1.5 and xmin=(α−1)/α=1/3; mean 1, infinite variance.',
                shift='Same light-tailed history and first 75 test days; then 3 × Pareto innovations for 75 days.',
                architecture={'Direct AR':'24 features → StandardScaler → Ridge (alpha 1 or 100) → 24 direct outputs; 576 coefficients + 24 intercepts.',
                    'Random Forest':'24 features → 100 multi-output trees → mean of 24-value leaves; bootstrap=True; min_samples_leaf 3 or 10; max_features=0.8; unlimited depth.',
                    'Extra Trees':'24 features → 100 multi-output trees with random split thresholds → mean of 24-value leaves; bootstrap=False; min_samples_leaf 3 or 10; max_features=0.8; unlimited depth.',
                    'XGBoost':'24 features → 100 boosting rounds per output, one_output_per_tree → 24 outputs; hist; max_depth 2 or 4; learning_rate=0.04; reg_lambda=10; subsample=0.85; colsample_bytree=0.85.',
                    'Repeat yesterday':'Copy all 24 previous-day observations; no fitting.'}),runs=[])
    for seed in SEEDS:
        worlds=generate(seed)
        # Same pre-change training data: reuse fits for light and shift.
        fit_cache={key:fit_models(worlds[key]) for key in ('light','pareto')}
        for key,days in worlds.items():
            x,y=days[:-1],days[1:]
            models,details=fit_cache['light' if key=='shift' else key]
            predictions={name:np.maximum(0,model.predict(x[TRAIN:])) for name,model in models.items()}
            predictions['Repeat yesterday']=x[TRAIN:]
            observed=y[TRAIN:]
            rows=[]
            for name,pred in predictions.items():
                row=dict(model=name,**metrics(observed,pred))
                if name!='Repeat yesterday':
                    row.update(next(d for d in details if d['model']==name))
                rows.append(row)
            selected=min(details,key=lambda d:d['validation_mae'])['model']
            report['runs'].append(dict(seed=seed,scenario=key,selected=selected,metrics=rows,
                actual=observed.tolist(),predictions={k:v.tolist() for k,v in predictions.items()},
                historical_q99=float(np.quantile(y[:TRAIN],.99)),historical_mean=float(y[:TRAIN].mean())))
            print(f'Seed {seed} / {key}: selected {selected}; MAE '+
                  ', '.join(f'{r["model"]}={r["mae"]:.3f}' for r in rows),flush=True)
    report['elapsed_seconds']=time.perf_counter()-begun
    out=ROOT/'docs'/'risk'
    out.mkdir(parents=True,exist_ok=True)
    (out/'data.js').write_text('window.RISK_DATA = '+json.dumps(report)+';',encoding='utf-8')
    (ROOT/'results'/'risk_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Finished in {report["elapsed_seconds"]:.1f} seconds',flush=True)


if __name__=='__main__':
    run()
