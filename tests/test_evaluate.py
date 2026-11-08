import threadcap  # noqa: F401
import unittest

import numpy as np

from evaluate import benign_majority_indices


class BenignMajorityTests(unittest.TestCase):
    def test_all_benign_and_uniform_malicious_are_shared_reproducibly(self):
        y = np.array([0] * 990 + [1] * 1000)
        first = benign_majority_indices(y)
        second = benign_majority_indices(y)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), len(first))
        self.assertEqual((y[first] == 0).sum(), 990)
        self.assertEqual((y[first] == 1).sum(), 10)
        self.assertLessEqual(y[first].mean(), 0.01)
        self.assertFalse(np.array_equal(first, benign_majority_indices(y, seed=43)))

    def test_no_oversampling_and_too_small_test_is_unavailable(self):
        y = np.array([0] * 990 + [1, 1])
        np.testing.assert_array_equal(benign_majority_indices(y), np.arange(len(y)))
        for labels in ([0] * 98 + [1], [0] * 100, [1] * 100, []):
            self.assertIsNone(benign_majority_indices(labels))

    def test_invalid_inputs_are_rejected(self):
        for ratio in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                benign_majority_indices([0, 1], ratio=ratio)
        for labels in ([[0, 1]], [0, 2]):
            with self.assertRaises(ValueError):
                benign_majority_indices(labels)


if __name__ == "__main__":
    unittest.main()
