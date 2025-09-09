#!/usr/bin/env python3
import argparse
from contextlib import asynccontextmanager
from io import BytesIO
import logging
import json
import os
import sys
import tempfile
import time
import warnings

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import soundfile as sf
import uvicorn

from indextts.infer_v2 import IndexTTS2
from tools.utils import eq, loudnorm
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 设置当前目录和路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

# FastAPI 相关导入
device = f"cuda:{int(os.getenv('TASK_SLOT'))-1}" if os.getenv("TASK_SLOT") else None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("index-tts-api")

# 过滤健康检查和文档页面的日志
logging.getLogger("uvicorn.access").addFilter(
    lambda r: "/health" not in r.getMessage() and "/docs" not in r.getMessage()
)

parser = argparse.ArgumentParser(description="IndexTTS API")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 to reduce memory and speed up on CUDA")
parser.add_argument("--use_cuda_kernel", action="store_true", default=False, help="Use BigVGAN custom CUDA kernel (CUDA only)")
cmd_args = parser.parse_args()

# 检查模型目录是否存在
if not os.path.exists(cmd_args.model_dir):
    logger.error(f"模型目录 {cmd_args.model_dir} 不存在，请先下载模型。")
    sys.exit(1)

# 全局模型实例
tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用程序生命周期管理
    """
    global tts
    
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)
    
    logger.info(
        "正在初始化 IndexTTS2 模型... (device=%s, fp16=%s, use_cuda_kernel=%s)",
        device,
        cmd_args.fp16,
        cmd_args.use_cuda_kernel,
    )
    try:
        tts = IndexTTS2(
            cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
            model_dir=cmd_args.model_dir,
            is_fp16=bool(cmd_args.fp16),
            use_cuda_kernel=bool(cmd_args.use_cuda_kernel),
            device=device,
        )
        logger.info("IndexTTS2 模型初始化完成")
        yield
    finally:
        logger.info("正在关闭 IndexTTS2 模型...")
        tts = None

# 创建 FastAPI 应用
app = FastAPI(
    title="IndexTTS API",
    description="IndexTTS2 语音合成 API 服务",
    version="2.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API 根路径"""
    return {"message": "欢迎使用 IndexTTS API 服务"}

@app.get("/health")
async def health_check():
    """健康检查端点"""
    if tts is None:
        raise HTTPException(status_code=503, detail="模型未初始化")
    return {"status": "healthy"}

@app.post("/generate")
async def generate_audio(
    text: str = Form(...),
    prompt_speech: UploadFile = File(...),
    # v2 生成控制参数
    emo_audio_prompt: UploadFile | None = File(None),
    emo_alpha: float = Form(1.0),
    emo_vector: str | None = Form(None),  # JSON 数组字符串，如 "[0,0,0,0,0,0,0.45,0]"
    use_emo_text: bool = Form(False),
    emo_text: str | None = Form(None),
    use_random: bool = Form(False),
    interval_silence: int = Form(200),
    # 文本/采样参数（与 v1 基本一致，v2 也支持）
    max_text_tokens_per_sentence: int = Form(120),
    do_sample: bool = Form(True),
    top_p: float = Form(0.8),
    top_k: int = Form(30),
    temperature: float = Form(0.8),
    length_penalty: float = Form(0.0),
    num_beams: int = Form(3),
    repetition_penalty: float = Form(10.0),
    max_mel_tokens: int = Form(1500),
    postprocess: bool = Form(True),
):
    """
    语音生成 API（v2）

    使用提供的说话人参考音频生成新的语音，可选使用情感参考音频、情感向量或文本情感描述进行控制。

    参数：
        text: 要合成的文本
        prompt_speech: 说话人参考音频（必需）
        emo_audio_prompt: 情感参考音频（可选）
        emo_alpha: 情感混合权重（默认 1.0）
        emo_vector: 情感向量 JSON 字符串（长度为8的数组，可选）
        use_emo_text: 是否使用文本描述提取情感（默认 False）
        emo_text: 文本情感描述（可选；use_emo_text=True 时有效）
        use_random: 情感向量随机化（默认 False）
        interval_silence: 句间静音时长（毫秒，默认 200）
        max_text_tokens_per_sentence: 分句的最大 token 数（默认 120）
        do_sample/top_p/top_k/temperature/length_penalty/num_beams/repetition_penalty/max_mel_tokens: 采样与长度控制参数
        postprocess: 是否进行响度归一化和EQ后处理（默认 True）

    返回：
        二进制 WAV 格式音频数据
    """
    if tts is None:
        raise HTTPException(status_code=503, detail="模型未初始化")
    
    try:
        # 处理参考音频
        temp_path = None
        emo_temp_path = None
        try:
            # 读取上传的音频文件
            contents = await prompt_speech.read()

            # 使用临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(contents)

            if emo_audio_prompt is not None:
                emo_contents = await emo_audio_prompt.read()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as emo_file:
                    emo_temp_path = emo_file.name
                    emo_file.write(emo_contents)

            # 解析 emo_vector（如果提供）
            emo_vector_list = None
            if emo_vector:
                try:
                    parsed = json.loads(emo_vector)
                    if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
                        emo_vector_list = [float(x) for x in parsed]
                except Exception:
                    logger.warning("emo_vector 解析失败，已忽略")

            # 准备推理参数
            kwargs = {
                "do_sample": bool(do_sample),
                "top_p": float(top_p),
                "top_k": int(top_k) if int(top_k) > 0 else None,
                "temperature": float(temperature),
                "length_penalty": float(length_penalty),
                "num_beams": num_beams,
                "repetition_penalty": float(repetition_penalty),
                "max_mel_tokens": int(max_mel_tokens),
            }

            # 设置输出路径
            output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

            # 执行推理（v2）
            logger.info(f"开始生成语音（v2），文本长度: {len(text)}")
            wav_path = tts.infer(
                spk_audio_prompt=temp_path,
                text=text,
                output_path=output_path,
                emo_audio_prompt=emo_temp_path,
                emo_alpha=float(emo_alpha),
                emo_vector=emo_vector_list,
                use_emo_text=bool(use_emo_text),
                emo_text=emo_text,
                use_random=bool(use_random),
                interval_silence=int(interval_silence),
                verbose=cmd_args.verbose,
                max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                **kwargs,
            )

            logger.info(f"语音生成完成，保存到: {wav_path}")

            # 读取生成的音频文件
            wav, sr = sf.read(wav_path, dtype='float32')

            # 后处理
            if postprocess:
                wav, _ = loudnorm(wav, sr)
                wav = eq(wav, sr)

            # 将音频数据转换为 WAV 格式的二进制数据
            buffer = BytesIO()
            sf.write(buffer, wav, sr, format='WAV')
            buffer.seek(0)

            # 返回二进制音频数据
            return Response(
                content=buffer.read(),
                media_type="audio/wav"
            )
        finally:
            # 确保临时文件被删除
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if emo_temp_path and os.path.exists(emo_temp_path):
                os.remove(emo_temp_path)
    except Exception as e:
        logger.exception(f"语音生成处理错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"语音生成处理错误: {str(e)}")


if __name__ == "__main__":
    logger.info(f"启动 IndexTTS API 服务，端口: {cmd_args.port}，主机: {cmd_args.host}")
    uvicorn.run(
        app,
        host=cmd_args.host,
        port=cmd_args.port,
        log_level="info"
    )
