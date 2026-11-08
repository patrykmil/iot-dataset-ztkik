import threadcap  # noqa: F401
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import config as C
from data import FEATURES, behavior_quotas, capture_batches, make_split, sample_capture, validate_split


def available_captures():
    return [c for c in C.CAPTURES if c != C.GENERALIZATION_BENIGN] + [
        C.GENERALIZATION_BENIGN + "__Somfy-01", C.GENERALIZATION_BENIGN + "__Somfy-02",
    ]


def feature_frame(labels):
    frame = pd.DataFrame({col: np.zeros(len(labels), dtype=np.float32) for col in C.NUMERIC})
    for col in C.CATEGORICAL:
        frame[col] = "tcp"
    frame[C.LABEL] = np.asarray(labels, dtype=np.int8)
    frame[C.DETAILED_LABEL] = np.where(frame[C.LABEL] == 1, "DDoS", "-")
    return frame[FEATURES + [C.LABEL, C.DETAILED_LABEL]]


class SplitTests(unittest.TestCase):
    def test_validation_has_separate_ddos_scanning_and_cc_captures(self):
        split = make_split(available_captures())
        for suffix in ("35-1", "48-1", "44-1"):
            capture = "CTU-IoT-Malware-Capture-" + suffix
            self.assertIn(capture, split["valid"])
            self.assertNotIn(capture, split["train"])

    def test_whole_family_and_somfy_parts_are_held_out(self):
        split = make_split(available_captures())
        for suffix in ("20-1", "21-1", "60-1"):
            self.assertIn("CTU-IoT-Malware-Capture-" + suffix, split["test_unknown"])
        for suffix in ("__Somfy-01", "__Somfy-02"):
            self.assertIn(C.GENERALIZATION_BENIGN + suffix, split["test_unknown"])
        self.assertEqual(split, make_split(reversed(available_captures())))

    def test_missing_required_capture_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            make_split([c for c in available_captures() if not c.endswith("21-1")])

    def test_family_leak_is_rejected(self):
        split = make_split(available_captures())
        capture = "CTU-IoT-Malware-Capture-21-1"
        split["test_unknown"].remove(capture)
        split["train"].append(capture)
        with self.assertRaisesRegex(ValueError, "rodziny"):
            validate_split(split)

    def test_somfy_parts_cannot_cross_subsets(self):
        split = make_split(available_captures())
        capture = C.GENERALIZATION_BENIGN + "__Somfy-02"
        split["test_unknown"].remove(capture)
        split["valid"].append(capture)
        with self.assertRaisesRegex(ValueError, "scenariuszy"):
            validate_split(split)


class SamplingTests(unittest.TestCase):
    def test_rare_class_survives_and_sampling_is_reproducible(self):
        frame = feature_frame([0] * 99 + [1])
        frame["duration"] = np.arange(100, dtype=np.float32)
        batches = [frame.iloc[:40].copy(), frame.iloc[40:].copy()]
        with patch("data.capture_batches", side_effect=lambda _, **kwargs: (b.copy() for b in batches)):
            first, info = sample_capture("fixture", 2)
            second, second_info = sample_capture("fixture", 2)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(info, second_info)
        self.assertEqual(len(first), 2)
        self.assertEqual(set(first[C.LABEL]), {0, 1})
        self.assertEqual(info["source_counts"], {"benign": 99, "malicious": 1})

    def test_small_and_single_class_capture(self):
        frame = feature_frame([1, 1, 1])
        with patch("data.capture_batches", side_effect=lambda _, **kwargs: iter([frame.copy()])):
            sample, info = sample_capture("fixture", 10)
        self.assertEqual(len(sample), 3)
        self.assertEqual(info["sample_counts"], {"benign": 0, "malicious": 3})

    def test_stratification_and_batch_boundaries(self):
        frame = feature_frame([0] * 75 + [1] * 25)
        frame["duration"] = np.arange(100, dtype=np.float32)
        with patch("data.capture_batches", side_effect=lambda _, **kwargs: iter([frame.copy()])):
            first, first_info = sample_capture("fixture", 20, balance=False)
        with patch("data.capture_batches", side_effect=lambda _, **kwargs: (frame.iloc[start:start + 7].copy() for start in range(0, 100, 7))):
            second, second_info = sample_capture("fixture", 20, balance=False)
        self.assertEqual(first_info["sample_counts"], {"benign": 15, "malicious": 5})
        self.assertEqual(first_info, second_info)
        pd.testing.assert_frame_equal(first, second)

    def test_class_and_behavior_limits_preserve_rare_rows(self):
        frame = feature_frame([0] * 4 + [1] * 1011)
        frame.loc[4:4, C.DETAILED_LABEL] = "FileDownload"
        frame.loc[5:14, C.DETAILED_LABEL] = "C&C"
        frame["duration"] = np.arange(len(frame), dtype=np.float32)
        with patch("data.capture_batches", side_effect=lambda _, **kwargs: iter([frame.copy()])):
            first, info = sample_capture("fixture", 20)
        with patch("data.capture_batches", side_effect=lambda _, **kwargs: (frame.iloc[start:start + 7].copy() for start in range(0, len(frame), 7))):
            second, second_info = sample_capture("fixture", 20)
        self.assertEqual(info["sample_counts"], {"benign": 4, "malicious": 10})
        self.assertEqual(info["sample_detailed_counts"], {"FileDownload": 1, "C&C": 4, "DDoS": 5})
        self.assertEqual(info, second_info)
        pd.testing.assert_frame_equal(first, second)
        self.assertNotIn(C.DETAILED_LABEL, FEATURES)

    def test_insufficient_behavior_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rodzajów ataku"):
            behavior_quotas({"C&C": 5, "DDoS": 100}, 1)

    def test_training_rejects_old_parquet_without_details_but_evaluation_can_read_it(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(C, "PARQUET_DIR", Path(directory)):
            feature_frame([0, 1]).drop(columns=C.DETAILED_LABEL).to_parquet(Path(directory) / "fixture.parquet", index=False)
            self.assertEqual(sum(len(batch) for batch in capture_batches("fixture")), 2)
            with self.assertRaisesRegex(ValueError, "detailed-label"):
                sample_capture("fixture", 10)

    def test_old_parquet_missing_ports_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(C, "PARQUET_DIR", Path(directory)):
            feature_frame([0, 1]).drop(columns="orig_p").to_parquet(Path(directory) / "fixture.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "orig_p"):
                list(capture_batches("fixture"))

    def test_invalid_numeric_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(C, "PARQUET_DIR", Path(directory)):
            frame = feature_frame([0, 1])
            frame.loc[0, "resp_p"] = 70000
            frame.to_parquet(Path(directory) / "fixture.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "port"):
                list(capture_batches("fixture"))


if __name__ == "__main__":
    unittest.main()
