from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn
import os
import wave
import time
from datetime import datetime

try:
    from .luna_core import LunaBrain
except ImportError:
    from luna_core import LunaBrain

app = FastAPI()
_luna = None

VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(VOICE_DIR, "audios")
if not os.path.exists(RECORD_DIR) :
    os.makedirs(RECORD_DIR)


def get_luna():
    global _luna
    if _luna is None:
        _luna = LunaBrain()
    return _luna


def get_pure_pcm(wav_path) :
    """确保提取的是纯 PCM 字节，跳过 WAV 头"""
    try :
        # 给硬盘一点点写入缓冲时间
        for _ in range(5) :
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 100 :
                break
            time.sleep(0.1)

        with wave.open(wav_path, 'rb') as wf :
            return wf.readframes(wf.getnframes())
    except Exception as e :
        print(f">>> 提取 PCM 失败: {e}")
        return None


@app.post("/upload_audio")
async def upload_audio(request: Request) :
    try :
        audio_bytes = await request.body()
        if not audio_bytes :
            return JSONResponse(status_code=400, content={"message" : "No data received"})

        file_id = datetime.now().strftime("%H%M%S")
        # 确保 luna_core 返回的是转换后的 .wav 文件名
        reply_filename = await get_luna().process_pipeline(audio_bytes, file_id)

        if reply_filename == "none" :
            return {"status" : "error", "file" : "none"}

        print(f">>> 流程结束，下发给ESP32: {reply_filename}")
        return {"status" : "success", "file" : reply_filename}
    except Exception as e :
        print(f">>> 服务器处理异常: {e}")
        return JSONResponse(status_code=500, content={"message" : str(e)})


@app.get("/get_audio/{filename}")
async def get_audio(filename: str) :
    file_path = os.path.join(RECORD_DIR, filename)

    # 核心修复：如果是请求 wav，我们剥离头部发 PCM
    if filename.endswith(".wav") :
        pcm_data = get_pure_pcm(file_path)
        if pcm_data :
            print(f">>> 成功发送纯 PCM 流: {filename} ({len(pcm_data)} bytes)")
            return Response(content=pcm_data, media_type="application/octet-stream")

    # 如果找不到或者不是 wav，尝试直接发（兜底）
    if os.path.exists(file_path) :
        with open(file_path, "rb") as f :
            return Response(content=f.read(), media_type="application/octet-stream")

    return Response(status_code=404)


if __name__ == "__main__" :
    print(">>> Luna 智能语音服务器启动中...")
    host = os.getenv("LUNACANE_VOICE_HOST", "0.0.0.0")
    port = int(os.getenv("LUNACANE_VOICE_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
