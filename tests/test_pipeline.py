from io import StringIO
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.train import daily_pairs, load_data, scores


class PipelineTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({'Date_time': pd.date_range('2020-01-01', periods=96, freq='h'),
                             'RS': np.arange(96, dtype=float)})

    def load(self, frame):
        return load_data(StringIO(frame.to_csv(index=False)))

    def test_calendar_alignment_and_boundary(self):
        x, y, dates = daily_pairs(self.load(self.fixture()))
        np.testing.assert_array_equal(x[0], np.arange(24))
        np.testing.assert_array_equal(y[0], np.arange(24, 48))
        self.assertEqual(dates[0], pd.Timestamp('2020-01-02'))
        # At a split, validation inputs may reuse known training observations,
        # but validation targets never occur in training targets.
        self.assertLess(y[0].max(), y[1].min())

    def test_missing_values_require_owner_decision(self):
        frame = self.fixture(); frame.loc[5, 'RS'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Ask the data owner'):
            self.load(frame)

    def test_missing_timestamp_requires_owner_decision(self):
        with self.assertRaisesRegex(ValueError, 'Missing hourly'):
            self.load(self.fixture().drop(index=5))

    def test_duplicate_rejected(self):
        frame = self.fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.load(pd.concat([frame, frame.iloc[:1]]))

    def test_metrics_do_not_cancel_signed_errors(self):
        result = scores(np.array([[10., 20.]]), np.array([[15., 15.]]))
        self.assertEqual(result['mae'], 5)
        self.assertEqual(result['bias'], 0)


if __name__ == '__main__':
    unittest.main()
