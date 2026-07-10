import importlib
import asyncio
import os
import tempfile
import unittest
from unittest import mock


VOICE_ENV_KEYS = (
    "LUNACANE_ALIYUN_NLS_APP_KEY",
    "LUNACANE_LLM_API_KEY",
    "LUNACANE_ALIYUN_AK_ID",
    "LUNACANE_ALIYUN_AK_SECRET",
    "LUNACANE_FFMPEG_PATH",
)


class VoiceImportTests(unittest.TestCase):
    def test_voice_modules_import_without_credentials(self):
        previous = {key: os.environ.pop(key, None) for key in VOICE_ENV_KEYS}
        try:
            luna_core = importlib.import_module("voice.luna_core")
            main_server = importlib.import_module("voice.main_server")
        finally:
            for key, value in previous.items():
                if value is not None:
                    os.environ[key] = value

        self.assertEqual(luna_core.LunaBrain.__name__, "LunaBrain")
        self.assertTrue(callable(main_server.get_luna))

    def test_audio_filename_validation_rejects_path_traversal(self):
        main_server = importlib.import_module("voice.main_server")

        self.assertTrue(main_server.is_safe_audio_filename("reply_20260708_120000_abcd1234.wav"))
        self.assertFalse(main_server.is_safe_audio_filename("../reply_20260708.wav"))
        self.assertFalse(main_server.is_safe_audio_filename("..\\reply_20260708.wav"))
        self.assertFalse(main_server.is_safe_audio_filename("other.wav"))
        self.assertFalse(main_server.is_safe_audio_filename("reply_20260708.mp3"))

    def test_reply_file_ids_are_unique(self):
        main_server = importlib.import_module("voice.main_server")

        ids = {main_server.make_reply_file_id() for _ in range(5)}

        self.assertEqual(len(ids), 5)
        for file_id in ids:
            self.assertRegex(file_id, r"^\d{8}_\d{6}_\d{6}_[0-9a-f]{8}$")

    def test_tts_temp_file_is_removed_on_conversion_failure(self):
        luna_core = importlib.import_module("voice.luna_core")

        class FakeCommunicate:
            def __init__(self, text, voice):
                self.text = text
                self.voice = voice

            async def save(self, path):
                with open(path, "wb") as f:
                    f.write(b"fake mp3")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "reply_test.wav")
            brain = object.__new__(luna_core.LunaBrain)

            with mock.patch.object(luna_core.edge_tts, "Communicate", FakeCommunicate), mock.patch.object(
                luna_core.subprocess,
                "run",
                side_effect=RuntimeError("ffmpeg failed"),
            ), mock.patch("builtins.print"):
                success = asyncio.run(brain.tts_to_wav("你好", out_file))

            self.assertFalse(success)
            self.assertFalse(os.path.exists(out_file + ".mp3"))


if __name__ == "__main__":
    unittest.main()
