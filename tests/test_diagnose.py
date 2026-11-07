import threadcap  # noqa: F401
import json
import unittest
from unittest.mock import patch

import numpy as np

import config as C
from diagnose import distribution, main, scores
from test_data import feature_frame


class DiagnosticTests(unittest.TestCase):
    def test_output_cannot_be_inside_artifacts(self):
        with patch("sys.argv", ["diagnose.py", str(C.ART / "lightgbm" / "diagnostics")]):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)

    def test_single_class_distribution_is_valid_json(self):
        frame = feature_frame([0, 0])
        y = frame.pop(C.LABEL).to_numpy()
        result = distribution(frame, y)
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["1"]["rows"], 0)
        self.assertEqual(result["1"]["numeric_quantiles"]["duration"], [])

    def test_constant_scores_reveal_prevalence_baseline(self):
        result = scores(np.array([0, 1, 1, 1]), np.full(4, 0.9), 1.01)
        self.assertEqual(result["unique_scores"], 1)
        self.assertEqual(result["roc_auc"], 0.5)
        self.assertEqual(result["average_precision"], 0.75)
        self.assertEqual(result["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
