import threadcap  # noqa: F401
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import config as C
from featurize import SOURCE_NUMERIC, capture_name, convert, main, parse_block, parse_zeek

CAPTURE = "CTU-IoT-Malware-Capture-34-1"
FIELDS = ["id.orig_p" if c == "orig_p" else "id.resp_p" if c == "resp_p" else c for c in C.CATEGORICAL + SOURCE_NUMERIC] + ["label", "detailed-label"]


def zeek_row(label="Benign", suffix_spaces=False):
    row = ["tcp", "-", "SF", "ShADad"] + ["1"] * len(SOURCE_NUMERIC)
    row[FIELDS.index("id.orig_p")] = "12345"
    row[FIELDS.index("id.resp_p")] = "443"
    if suffix_spaces:
        return "\t".join(row) + f"   {label}   -\n"
    return "\t".join(row + [label, "-"]) + "\n"


def zeek_text(rows):
    return "#separator \\x09\n#fields\t" + "\t".join(FIELDS) + "\n" + "".join(rows)


class ParserTests(unittest.TestCase):
    def test_ports_labels_and_appended_annotations(self):
        frame = parse_block([zeek_row(), zeek_row("Malicious", True), zeek_row("-")], FIELDS, "\t", CAPTURE)
        self.assertEqual(frame[C.LABEL].tolist(), [0, 1])
        self.assertEqual(frame["orig_p"].tolist(), [12345, 12345])
        self.assertEqual(frame["resp_p"].tolist(), [443, 443])
        self.assertEqual(frame["bytes_total"].tolist(), [2, 2])
        self.assertNotIn("detailed-label", frame)

    def test_missing_header_label_and_malformed_rows_are_errors(self):
        with self.assertRaisesRegex(ValueError, "label"):
            parse_block([], FIELDS[:-2], "\t", CAPTURE)
        with self.assertRaisesRegex(ValueError, "pól"):
            parse_block(["broken\n"], FIELDS, "\t", CAPTURE)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "conn.log.labeled"
            source.write_text(zeek_row())
            with self.assertRaisesRegex(ValueError, "przed #fields"):
                list(parse_zeek(source, CAPTURE))

    def test_invalid_ports_and_zero_denominators(self):
        values = zeek_row().rstrip().split("\t")
        values[FIELDS.index("id.resp_p")] = "70000"
        with self.assertRaisesRegex(ValueError, "port"):
            parse_block(["\t".join(values)], FIELDS, "\t", CAPTURE)
        values[FIELDS.index("id.resp_p")] = "443"
        values[FIELDS.index("orig_pkts")] = "0"
        values[FIELDS.index("orig_bytes")] = "0"
        frame = parse_block(["\t".join(values)], FIELDS, "\t", CAPTURE)
        self.assertEqual(frame["bytes_per_pkt"].iloc[0], 0)
        self.assertEqual(frame["resp_orig_ratio"].iloc[0], 1)

    def test_streaming_conversion_counts_and_failure_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "conn.log.labeled", Path(directory) / "capture.parquet"
            source.write_text(zeek_text([zeek_row(), zeek_row("Malicious"), zeek_row("unknown")]))
            info = convert(source, CAPTURE, output)
            self.assertEqual(info["flows"], 2)
            self.assertEqual(info["dropped_labels"], 1)
            self.assertEqual(info["benign"], 1)
            self.assertEqual(info["malicious"], 1)
            previous = output.read_bytes()
            source.write_text("broken\n")
            with self.assertRaises(ValueError):
                convert(source, CAPTURE, output)
            self.assertEqual(previous, output.read_bytes())
            self.assertFalse(output.with_name(output.name + ".tmp").exists())

    def test_chunks_and_limit_are_source_row_based(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "conn.log.labeled"
            source.write_text(zeek_text([zeek_row(), zeek_row("Malicious"), zeek_row()]))
            chunks = list(parse_zeek(source, CAPTURE, limit=2, block=1))
            self.assertEqual([len(frame) for frame, _ in chunks], [1, 1])
            self.assertEqual(sum(read for _, read in chunks), 2)

    def test_capture_names_for_both_somfy_layouts(self):
        base = Path("/example") / C.GENERALIZATION_BENIGN
        expected = C.GENERALIZATION_BENIGN + "__Somfy-01"
        self.assertEqual(capture_name(base / "Somfy-01" / "bro" / "conn.log.labeled"), expected)
        self.assertEqual(capture_name(base / f"{expected}.conn.log.labeled"), expected)
        self.assertEqual(capture_name(Path("/example") / CAPTURE / "bro" / "conn.log.labeled"), CAPTURE)

    def test_partial_conversion_merges_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / CAPTURE / "conn.log.labeled"
            source.parent.mkdir()
            source.write_text(zeek_text([zeek_row()]))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"existing": {"flows": 42}}))
            with patch.object(C, "MANIFEST_PATH", manifest), patch.object(C, "PARQUET_DIR", root / "parquet"), patch("sys.argv", ["featurize.py", str(source)]):
                main()
            result = json.loads(manifest.read_text())
            self.assertEqual(result["existing"]["flows"], 42)
            self.assertEqual(result[CAPTURE]["flows"], 1)
            self.assertEqual(len(pd.read_parquet(root / "parquet" / f"{CAPTURE}.parquet")), 1)


if __name__ == "__main__":
    unittest.main()
