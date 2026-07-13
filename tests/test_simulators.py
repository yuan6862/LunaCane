import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import simulate_imu, simulate_voice


class FakeResponse:
    def __init__(self, payload=None, content=b"", status_code=200):
        self.payload = payload or {}
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self.payload


class SimulatorTests(unittest.TestCase):
    def test_imu_csv_is_batched_like_firmware(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "imu.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=simulate_imu.SENSOR_FIELDS)
                writer.writeheader()
                for index in range(45):
                    writer.writerow({field: index for field in simulate_imu.SENSOR_FIELDS})

            samples = simulate_imu.load_samples(path)
            with mock.patch.object(
                simulate_imu.requests, "post", return_value=FakeResponse({"status": "ok"})
            ) as post:
                sent = simulate_imu.replay(samples, "http://collector/sensor", realtime=False)

        self.assertEqual(sent, 45)
        self.assertEqual([len(call.kwargs["json"]["samples"]) for call in post.call_args_list], [20, 20, 5])
        self.assertEqual(samples[1]["t"], 20)

    def test_voice_simulator_uploads_and_downloads_reply(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "input.wav"
            output = Path(tmpdir) / "reply.pcm"
            source.write_bytes(b"RIFF-test")
            with mock.patch.object(
                simulate_voice.requests,
                "post",
                return_value=FakeResponse({"status": "success", "file": "reply_test.wav"}),
            ), mock.patch.object(
                simulate_voice.requests, "get", return_value=FakeResponse(content=b"pcm")
            ):
                filename = simulate_voice.run(source, "http://voice", output)

            self.assertEqual(filename, "reply_test.wav")
            self.assertEqual(output.read_bytes(), b"pcm")


if __name__ == "__main__":
    unittest.main()
