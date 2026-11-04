from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import pandas as pd

import plot_verification


class PlotVerificationTests(unittest.TestCase):
    def test_plot_uses_display_name_and_preserves_traces_for_supported_csv_names(self):
        columns = ("imu_ax", "imu_ay", "imu_az", "imu_gx", "imu_gy", "imu_gz")
        clean_data = pd.DataFrame(
            {"time": [4.0, 4.5, 5.0], **{column: [1.0, 2.0, 3.0] for column in columns}}
        )
        realistic_data = pd.DataFrame(
            {"time": [4.0, 4.5, 5.0], **{column: [1.1, 2.1, 3.1] for column in columns}}
        )

        for prefix in ("realistic", "noisy", "gui"):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as temp_dir:
                folder = Path(temp_dir)
                clean_data.to_csv(folder / "clean_imu_test.csv", index=False)
                realistic_data.to_csv(folder / f"{prefix}_imu_test.csv", index=False)
                try:
                    with mock.patch.object(plot_verification.plt, "close"):
                        plot_verification.plot_verification(str(folder))
                        figure = plot_verification.plt.gcf()
                        self.assertEqual(len(figure.axes), 6)
                        for axis in figure.axes:
                            self.assertEqual(len(axis.lines), 2)
                            self.assertEqual(
                                axis.lines[0].get_label(), "Realistic Sim2Real IMU (ASM330LHHX)"
                            )
                            self.assertEqual(axis.lines[1].get_label(), "Clean (Isaac)")
                            self.assertEqual(list(axis.lines[0].get_xdata()), [0.0, 0.5, 1.0])
                            self.assertEqual(list(axis.lines[0].get_ydata()), [1.1, 2.1, 3.1])
                            self.assertEqual(list(axis.lines[1].get_ydata()), [1.0, 2.0, 3.0])
                        self.assertTrue(
                            (folder / "verification_plot.png").read_bytes().startswith(
                                b"\x89PNG\r\n\x1a\n"
                            )
                        )
                finally:
                    plot_verification.plt.close("all")


if __name__ == "__main__":
    unittest.main()