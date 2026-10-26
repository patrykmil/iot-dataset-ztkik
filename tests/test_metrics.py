import threadcap  # noqa: F401
import unittest

import numpy as np

from metrics import pick_threshold, score_metrics, threshold_rates


class ThresholdTests(unittest.TestCase):
    def test_ties_cannot_be_split_to_claim_lower_fpr(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.9, 0.8, 0.1])
        threshold, rates = pick_threshold(y, p, 0.0)
        self.assertGreater(threshold, p.max())
        self.assertEqual(rates["fpr"], 0.0)
        self.assertEqual(rates["recall"], 0.0)
        self.assertEqual(rates, threshold_rates(y, p, threshold))

    def test_returned_rates_match_selected_threshold(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.95, 0.9, 0.8, 0.1])
        threshold, rates = pick_threshold(y, p, 0.5)
        self.assertEqual(threshold, 0.8)
        self.assertEqual(rates["recall"], 1.0)
        self.assertEqual(rates["fpr"], 0.5)
        self.assertEqual(rates, threshold_rates(y, p, threshold))

    def test_all_one_scores_have_finite_no_alarm_threshold(self):
        threshold, rates = pick_threshold([0, 1], [1.0, 1.0], 0.01)
        self.assertTrue(np.isfinite(threshold))
        self.assertGreater(threshold, 1.0)
        self.assertEqual(rates["confusion_at_threshold"], [[1, 0], [1, 0]])

    def test_single_class_groups_still_report_operating_metrics(self):
        result = score_metrics([0, 0], [0.9, 0.1], 0.5)
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["average_precision"])
        self.assertIsNone(result["recall"])
        self.assertEqual(result["fpr"], 0.5)
        result = score_metrics([1, 1], [0.9, 0.1], 0.5)
        self.assertIsNone(result["fpr"])
        self.assertEqual(result["recall"], 0.5)

    def test_invalid_inputs_are_rejected(self):
        for y, p in (([0], [float("nan")]), ([0], [1.1]), ([2], [0.5]), ([], []), ([0, 1], [0.2])):
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                score_metrics(y, p, 0.5)
        with self.assertRaises(ValueError):
            pick_threshold([0, 0], [0.1, 0.2], 0.01)


if __name__ == "__main__":
    unittest.main()
