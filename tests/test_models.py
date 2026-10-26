import threadcap  # noqa: F401
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

import config as C
from models import categorical_frame, load_model, make_preprocessor, predict, train_logistic
from test_data import feature_frame


class PreprocessingTests(unittest.TestCase):
    def test_unknown_category_and_preprocessor_roundtrip(self):
        fit = feature_frame([0, 1, 0, 1]).drop(columns=C.LABEL)
        fit.loc[2:, "proto"] = "udp"
        valid = fit.iloc[:1].copy()
        valid["proto"] = "new-protocol"
        processor = make_preprocessor().fit(fit)
        encoded = processor.transform(valid)
        values = encoded.toarray() if hasattr(encoded, "toarray") else encoded
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(encoded.shape[1], processor.transform(fit).shape[1])
        np.testing.assert_allclose(processor.named_transformers_["counts"].named_steps["standardscaler"].mean_, 0)
        categorical = categorical_frame(valid, {col: fit[col].unique().tolist() for col in C.CATEGORICAL})
        self.assertTrue(categorical["proto"].isna().all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processor.joblib"
            joblib.dump(processor, path)
            restored = joblib.load(path).transform(valid)
            np.testing.assert_allclose(values, restored.toarray() if hasattr(restored, "toarray") else restored)

    def test_logistic_model_roundtrip(self):
        frame = feature_frame([0, 1] * 20)
        frame["duration"] = frame[C.LABEL].astype(np.float32) * 10
        y = frame.pop(C.LABEL).to_numpy()
        processor = make_preprocessor()
        X = processor.fit_transform(frame).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory, patch.object(C, "LOGISTIC_EPOCHS", 2):
            model, metadata = train_logistic(X, y, X, y, Path(directory))
            expected = predict("logistic", model, frame, metadata, processor)
            loaded = load_model("logistic", Path(directory), metadata, "cpu")
            np.testing.assert_allclose(expected, predict("logistic", loaded, frame, metadata, processor))
            self.assertTrue(((expected >= 0) & (expected <= 1)).all())


if __name__ == "__main__":
    unittest.main()
