"""Mały CPU smoke test całego porównania; nie wymaga datasetu ani GPU."""

import threadcap  # noqa: F401
import importlib.util
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import config as C
from test_data import feature_frame


@unittest.skipUnless(importlib.util.find_spec("lightgbm") and importlib.util.find_spec("torch"), "wymaga LightGBM i PyTorch")
class PipelineTests(unittest.TestCase):
    def test_all_three_models_train_reload_and_compare(self):
        import evaluate
        import train

        captures = [
            "CTU-IoT-Malware-Capture-43-1",
            *C.VALID_CAPTURES, *C.KNOWN_TEST_CAPTURES,
            "CTU-IoT-Malware-Capture-20-1", "CTU-IoT-Malware-Capture-21-1",
            "CTU-IoT-Malware-Capture-60-1", C.GENERALIZATION_BENIGN + "__Somfy-01",
        ]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            parquet = root / "parquet"
            parquet.mkdir()
            manifest = {}
            for capture in captures:
                labels = [0] * 40 if "Honeypot" in capture else [0, 1] * 20
                frame = feature_frame(labels)
                frame["duration"] = np.asarray(labels, dtype=np.float32) * 10
                frame["orig_bytes"] = np.asarray(labels, dtype=np.float32) * 100
                frame.to_parquet(parquet / f"{capture}.parquet", index=False)
                manifest[capture] = {"flows": len(frame), "limited": False}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            overrides = {
                "ART": root, "PARQUET_DIR": parquet, "MANIFEST_PATH": manifest_path,
                "EXPERIMENT_PATH": root / "experiment.json",
                "LGBM_ROUNDS": 3, "LGBM_EARLY_STOP": 2,
                "LGBM_PARAMS": {**C.LGBM_PARAMS, "num_leaves": 4, "min_data_in_leaf": 1},
                "LOGISTIC_EPOCHS": 2, "MLP_EPOCHS": 2,
            }
            for key, value in overrides.items():
                stack.enter_context(patch.object(C, key, value))
            with patch("sys.argv", ["train.py", "--device", "cpu", "--cap-per-capture", "30"]):
                train.main()
            experiment = json.loads(C.EXPERIMENT_PATH.read_text())
            self.assertEqual(experiment["status"], "complete")
            self.assertEqual(set(experiment["models"]), {"lightgbm", "logistic", "mlp"})
            with patch("sys.argv", ["evaluate.py", "--device", "cpu"]):
                evaluate.main()
            comparison = json.loads((root / "comparison.json").read_text())
            self.assertEqual(comparison["experiment_id"], experiment["id"])
            for name in experiment["models"]:
                metadata = json.loads((root / name / "metadata.json").read_text())
                self.assertLessEqual(metadata["validation"]["fpr"], C.TARGET_FPR)
                for subset in ("test_known", "test_unknown"):
                    metrics = comparison["models"][name][subset]
                    self.assertEqual(metrics["threshold"], metadata["threshold"])
                    self.assertEqual(metrics["n"], 40 * len(experiment["split"][subset]))
            self.assertEqual(len(pd.read_csv(root / "comparison.csv")), 6)
            self.assertTrue((root / "test_known_curves.png").exists())
            experiment["status"] = "running"
            C.EXPERIMENT_PATH.write_text(json.dumps(experiment))
            with patch("sys.argv", ["evaluate.py", "--device", "cpu"]), self.assertRaisesRegex(ValueError, "ukończony"):
                evaluate.main()


if __name__ == "__main__":
    unittest.main()
