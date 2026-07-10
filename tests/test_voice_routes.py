import asyncio
import os
import tempfile
import unittest
import wave
from unittest import mock

from voice import main_server


class FakeRequest:
    def __init__(self, body):
        self._body = body

    async def body(self):
        return self._body


class FakeLuna:
    async def process_pipeline(self, audio_bytes, file_id):
        self.audio_bytes = audio_bytes
        self.file_id = file_id
        return f"reply_{file_id}.wav"


class VoiceRouteTests(unittest.TestCase):
    def test_upload_audio_uses_mock_pipeline_without_credentials(self):
        luna = FakeLuna()
        with mock.patch.object(main_server, "get_luna", return_value=luna):
            result = asyncio.run(main_server.upload_audio(FakeRequest(b"RIFF-test")))

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["file"].startswith("reply_"))
        self.assertEqual(luna.audio_bytes, b"RIFF-test")

    def test_empty_upload_is_rejected(self):
        response = asyncio.run(main_server.upload_audio(FakeRequest(b"")))
        self.assertEqual(response.status_code, 400)

    def test_get_audio_returns_pcm_frames_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "reply_test.wav"
            path = os.path.join(tmpdir, filename)
            with wave.open(path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x01\x02\x03\x04")

            with mock.patch.object(main_server, "RECORD_DIR", tmpdir):
                response = asyncio.run(main_server.get_audio(filename))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body, b"\x01\x02\x03\x04")

    def test_get_audio_rejects_unsafe_name(self):
        response = asyncio.run(main_server.get_audio("other.wav"))
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
