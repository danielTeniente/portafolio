import unittest
import numpy as np
from src.risk_benchmark import generate, metrics, TRAIN, TEST


class RiskTests(unittest.TestCase):
    def test_control_and_shift_have_identical_history(self):
        worlds=generate(42)
        change=TRAIN+1+TEST//2
        np.testing.assert_array_equal(worlds['light'][:change],worlds['shift'][:change])
        np.testing.assert_array_equal(worlds['shift'][change:],3*worlds['pareto'][change:])
        self.assertEqual(worlds['light'].shape,(751,24))

    def test_seeds_are_reproducible_and_outputs_positive(self):
        first,second=generate(7),generate(7)
        for key in first:
            np.testing.assert_array_equal(first[key],second[key])
            self.assertTrue(np.isfinite(first[key]).all())
            self.assertTrue((first[key]>0).all())

    def test_underforecast_is_not_absolute_error(self):
        result=metrics(np.array([[1.,10.]]),np.array([[20.,2.]]))
        self.assertEqual(result['worst_underforecast'],8.)
        self.assertEqual(result['top_one_percent_underforecast'],8.)


if __name__=='__main__':
    unittest.main()
