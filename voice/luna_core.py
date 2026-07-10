import os
import re
import json
import shutil
import requests
from openai import OpenAI
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import edge_tts
import subprocess
from dotenv import load_dotenv

_VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_VOICE_DIR)
load_dotenv(os.path.join(_PROJECT_DIR, ".env"))

CONFIG = {
    "APP_KEY": os.getenv("LUNACANE_ALIYUN_NLS_APP_KEY", ""),
    "LLM_KEY": os.getenv("LUNACANE_LLM_API_KEY", ""),
    "LLM_BASE_URL": os.getenv("LUNACANE_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "LLM_MODEL": os.getenv("LUNACANE_LLM_MODEL", "qwen-plus"),
    "ALIYUN_AK_ID": os.getenv("LUNACANE_ALIYUN_AK_ID", ""),
    "ALIYUN_AK_SECRET": os.getenv("LUNACANE_ALIYUN_AK_SECRET", ""),
    "AUDIO_SAVE_DIR": os.path.join(_VOICE_DIR, "audios"),
    "FFMPEG_PATH": os.getenv("LUNACANE_FFMPEG_PATH") or shutil.which("ffmpeg") or "",
}

os.makedirs(CONFIG["AUDIO_SAVE_DIR"], exist_ok=True)

def clean_reply_text(text):
    if not text: return "我暂时没有想到合适的回答。"
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"[*#_~✨⭐🌟💛😊😄😆😉🤖☀️🌈🎉💡🔥❤️🧡💚💙💜]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class LunaBrain:
    def __init__(self):
        self._validate_config()
        self.llm = OpenAI(
            api_key=CONFIG["LLM_KEY"],
            base_url=CONFIG["LLM_BASE_URL"],
        )
        self.speech_token = self.get_aliyun_nls_token()

    def _validate_config(self):
        missing = [
            name for name in (
                "APP_KEY",
                "LLM_KEY",
                "ALIYUN_AK_ID",
                "ALIYUN_AK_SECRET",
                "FFMPEG_PATH",
            )
            if not CONFIG[name]
        ]
        if missing:
            raise RuntimeError(
                "Missing voice configuration: "
                + ", ".join(missing)
                + ". See .env.example."
            )

    def get_aliyun_nls_token(self):
        try:
            client = AcsClient(CONFIG["ALIYUN_AK_ID"], CONFIG["ALIYUN_AK_SECRET"], "cn-shanghai")
            request = CommonRequest()
            request.set_method("POST")
            request.set_domain("nls-meta.cn-shanghai.aliyuncs.com")
            request.set_version("2019-02-28")
            request.set_action_name("CreateToken")
            response = client.do_action_with_exception(request)
            jss = json.loads(response.decode("utf-8"))
            return jss["Token"]["Id"]
        except Exception as e:
            print(f">>> Token 获取异常: {e}")
            return None

    def aliyun_asr(self, audio_data):
        if not self.speech_token:
            self.speech_token = self.get_aliyun_nls_token()
        # 跳过 WAV 44字节头
        pcm_data = audio_data[44:] if len(audio_data) > 44 else audio_data
        url = f"http://nls-gateway.cn-shanghai.aliyuncs.com/stream/v1/asr?appkey={CONFIG['APP_KEY']}&format=pcm&sample_rate=16000"
        headers = {"X-NLS-Token": self.speech_token, "Content-Type": "application/octet-stream"}
        try:
            r = requests.post(url, headers=headers, data=pcm_data, timeout=10)
            return r.json().get("result")
        except Exception as e:
            print(f">>> ASR 异常: {e}"); return None

    def ask_qwen(self, text):
        try:
            response = self.llm.chat.completions.create(
                model=CONFIG["LLM_MODEL"],
                messages=[
                    {"role": "system", "content": "你是智能拐杖助手Luna。请用温和、简短的口语回答。不要使用符号，控制在30字内。"},
                    {"role": "user", "content": text}
                ]
            )
            return clean_reply_text(response.choices[0].message.content)
        except Exception as e:
            print(f">>> LLM 异常: {e}"); return "对不起，网络不太稳定。"

    async def tts_to_wav(self, text, out_file) :
        temp_mp3 = out_file + ".mp3"
        try :
            # 1. 生成临时 mp3 (edge-tts 吐出来的原始格式)
            communicate = edge_tts.Communicate(text=text, voice="zh-CN-XiaoxiaoNeural")
            await communicate.save(temp_mp3)

            command = [
                CONFIG["FFMPEG_PATH"], "-y", "-i", temp_mp3,
                "-ar", "16000", "-ac", "1", "-f", "wav", out_file
            ]

            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return True
        except Exception as e :
            print(f">>> 转换失败: {e}")
            return False
        finally :
            if os.path.exists(temp_mp3) :
                os.remove(temp_mp3)

    async def process_pipeline(self, audio_bytes, file_id):
        user_text = self.aliyun_asr(audio_bytes)
        if not user_text: return "none"
        print(f">>> 老人说: {user_text}")

        reply_text = self.ask_qwen(user_text)
        print(f">>> Luna答: {reply_text}")

        file_name = f"reply_{file_id}.wav"
        save_path = os.path.join(CONFIG["AUDIO_SAVE_DIR"], file_name)
        success = await self.tts_to_wav(reply_text, save_path)
        return file_name if success else "none"
