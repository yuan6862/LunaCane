import importlib
import os
import unittest


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


if __name__ == "__main__":
    unittest.main()
