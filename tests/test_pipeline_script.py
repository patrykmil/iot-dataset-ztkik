"""Sprawdza kolejność etapów i zatrzymanie przy błędzie, bez uruchamiania modeli."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class PipelineScriptTests(unittest.TestCase):
    def run_pipeline(self, directory, fail_stage="", raw_state="present"):
        root = Path(directory) / "project with spaces"
        root.mkdir()
        shutil.copyfile(Path(__file__).resolve().parents[1] / "pipeline.sh", root / "pipeline.sh")
        raw = root / "raw"
        if raw_state != "missing":
            raw.mkdir()
        if raw_state == "present":
            capture = raw / "capture"
            capture.mkdir()
            (capture / "conn.log.labeled").write_text("fixture\n")
        elif raw_state == "unrelated":
            (raw / "README.txt").write_text("not a dataset\n")
        elif raw_state == "zero":
            (raw / "conn.log.labeled").touch()
        python = root / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text(
            '#!/usr/bin/env bash\n'
            'printf "%s|%s\\n" "$DETECTOR_THREADS" "$*" >> "$PIPELINE_CALLS"\n'
            'if [[ "$2" == "$PIPELINE_FAIL_STAGE" ]]; then exit 7; fi\n'
        )
        python.chmod(0o755)
        calls = Path(directory) / "calls.txt"
        env = {key: value for key, value in os.environ.items() if key not in ("DETECTOR_THREADS", "DETECTOR_RAW")}
        env.update({"PIPELINE_CALLS": str(calls), "PIPELINE_FAIL_STAGE": fail_stage})
        result = subprocess.run(["bash", str(root / "pipeline.sh")], cwd=directory, env=env, capture_output=True, text=True)
        return result, calls.read_text().splitlines()

    def test_default_pipeline_runs_all_stages_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls = self.run_pipeline(directory)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["4|-u featurize.py raw", "4|-u train.py", "4|-u evaluate.py"])
        self.assertIn("Pipeline zakończony", result.stdout)

    def test_failure_stops_subsequent_stages_and_preserves_exit_code(self):
        for index, stage in enumerate(("featurize.py", "train.py", "evaluate.py"), start=1):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                result, calls = self.run_pipeline(directory, stage)
                self.assertEqual(result.returncode, 7)
                self.assertEqual(len(calls), index)
                self.assertNotIn("Pipeline zakończony", result.stdout)

    def test_missing_or_empty_raw_downloads_dataset_before_training(self):
        for state in ("missing", "empty", "unrelated", "zero"):
            with self.subTest(raw_state=state), tempfile.TemporaryDirectory() as directory:
                result, calls = self.run_pipeline(directory, raw_state=state)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(calls, ["4|-u featurize.py --download", "4|-u train.py", "4|-u evaluate.py"])

    def test_failed_download_stops_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls = self.run_pipeline(directory, fail_stage="featurize.py", raw_state="empty")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(calls, ["4|-u featurize.py --download"])
        self.assertNotIn("Pipeline zakończony", result.stdout)


if __name__ == "__main__":
    unittest.main()
