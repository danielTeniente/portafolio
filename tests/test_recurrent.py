import unittest
import numpy as np
from src.recurrent import ForecastNetwork, RecurrentRegressor
import torch


class RecurrentTests(unittest.TestCase):
    def test_all_architectures_produce_direct_day_profiles(self):
        for name in ('RNN', 'GRU', 'LSTM'):
            model = ForecastNetwork(name, 8, 13)
            self.assertEqual(tuple(model(torch.zeros(3, 13, 1)).shape), (3, 13))

    def test_epoch_selection_refits_scaling_on_training_only(self):
        x = np.arange(40*13, dtype=float).reshape(40, 13)
        y = x+10
        model = RecurrentRegressor('GRU', 8, max_epochs=3, patience=2).fit(x, y)
        self.assertEqual(model.inner_train_days_, 32)
        self.assertEqual(model.x_mean_, x.mean())
        self.assertTrue(1 <= model.epochs_ <= 3)
        before = model.x_mean_
        pred = model.predict(np.full((2,13), 9999.))
        self.assertEqual(model.x_mean_, before)
        self.assertTrue(np.isfinite(pred).all())

    def test_seed_and_fixed_epochs_reproduce_predictions(self):
        x = np.arange(20*13, dtype=float).reshape(20,13)
        y = x+10
        a = RecurrentRegressor('RNN',8).fit(x,y,epochs=2).predict(x)
        b = RecurrentRegressor('RNN',8).fit(x,y,epochs=2).predict(x)
        np.testing.assert_allclose(a,b)


if __name__ == '__main__':
    unittest.main()
