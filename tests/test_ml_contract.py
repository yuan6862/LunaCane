import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocess import (
    FEATURE_COLS,
    build_binary_classification_metrics,
    create_sliding_windows,
    write_deployment_metadata,
)


class DeploymentContractTests(unittest.TestCase):
    def test_metadata_and_header_share_the_same_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = write_deployment_metadata(tmpdir, 100, 50, 0.42)
            saved = json.loads((Path(tmpdir) / "model_metadata.json").read_text(encoding="utf-8"))
            header = (Path(tmpdir) / "model_config.h").read_text(encoding="utf-8")

        self.assertEqual(saved, metadata)
        self.assertEqual(metadata["feature_columns"], FEATURE_COLS)
        self.assertEqual(metadata["input_shape"], [1, 100, 8])
        self.assertIn("MODEL_FALL_THRESHOLD = 0.420000f", header)
        self.assertIn("MODEL_WINDOW_SIZE = 100", header)

    def test_metadata_records_custom_positive_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = write_deployment_metadata(
                tmpdir, 100, 50, 0.42, positive_label="sitdown"
            )

        self.assertEqual(metadata["positive_label"], "sit_down")

    def test_binary_metrics_keep_two_classes_for_single_class_test_set(self):
        report, matrix, printable_report = build_binary_classification_metrics(
            np.array([0, 0]), np.array([0, 1])
        )

        self.assertEqual(np.asarray(matrix).shape, (2, 2))
        self.assertIn("正常动作", report)
        self.assertIn("跌倒", report)
        self.assertIn("正常动作", printable_report)
        self.assertIn("跌倒", printable_report)

    def test_windows_never_cross_recording_boundaries(self):
        rows = []
        for source, value in (("session_a.csv", 1.0), ("session_b.csv", 9.0)):
            for _ in range(6):
                row = {feature: value for feature in FEATURE_COLS}
                row.update({"numeric_label": 0, "source_file": source})
                rows.append(row)

        X, _, groups = create_sliding_windows(
            pd.DataFrame(rows), window_size=4, step_size=2
        )

        self.assertEqual(groups.tolist(), ["session_a.csv", "session_a.csv", "session_b.csv", "session_b.csv"])
        for window in X:
            self.assertEqual(len(np.unique(window[:, 0])), 1)


if __name__ == "__main__":
    unittest.main()
