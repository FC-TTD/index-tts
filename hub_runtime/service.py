"""Shared existing Index API inference pipeline; UI calls the same function."""

import json
import logging
import os
import tempfile
from io import BytesIO
import librosa
import soundfile as sf
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from download_filename import build_content_disposition, build_download_filename
from ttd_model_runtime.audio.postprocess import apply_postprocess, trim_silence
from .parameters import DEFAULTS

logger = logging.getLogger(__name__)

LANGUAGE_ALIASES = {
    "zh": "ZH",
    "zh-cn": "ZH",
    "chinese": "ZH",
    "中文": "ZH",
    "en": "EN",
    "english": "EN",
    "英文": "EN",
    "ja": "JA",
    "jp": "JA",
    "japanese": "JA",
    "日文": "JA",
    "日语": "JA",
    "es": "ES",
    "spanish": "ES",
    "西班牙文": "ES",
    "西班牙语": "ES",
    "ar": "AR",
    "arabic": "AR",
    "阿拉伯文": "AR",
    "阿拉伯语": "AR",
}


def normalize_language(language: str | None) -> str:
    value = (language or "ZH").strip()
    normalized = LANGUAGE_ALIASES.get(value.lower(), value.upper())
    if normalized not in {"ZH", "EN", "JA", "ES", "AR"}:
        raise ValueError("language must be one of ZH, EN, JA, ES, AR")
    return normalized


def speed_to_duration_factor(speed: float) -> float:
    value = float(speed)
    return 1.0 / value if value > 0 else 1.0


def synthesize(
    runtime,
    args,
    text: str = Form(...),
    language: str = Form(DEFAULTS["language"]),
    prompt_speech: UploadFile = File(...),
    emo_audio_prompt: UploadFile | None = File(DEFAULTS["emo_audio_prompt"]),
    emo_alpha: float = Form(DEFAULTS["emo_alpha"]),
    emo_vector: str | None = Form(DEFAULTS["emo_vector"]),
    use_emo_text: bool = Form(DEFAULTS["use_emo_text"]),
    emo_text: str | None = Form(DEFAULTS["emo_text"]),
    use_random: bool = Form(DEFAULTS["use_random"]),
    interval_silence: int = Form(DEFAULTS["interval_silence"]),
    max_text_tokens_per_sentence: int = Form(DEFAULTS["max_text_tokens_per_sentence"]),
    do_sample: bool = Form(DEFAULTS["do_sample"]),
    top_p: float = Form(DEFAULTS["top_p"]),
    top_k: int = Form(DEFAULTS["top_k"]),
    temperature: float = Form(DEFAULTS["temperature"]),
    length_penalty: float = Form(DEFAULTS["length_penalty"]),
    num_beams: int = Form(DEFAULTS["num_beams"]),
    repetition_penalty: float = Form(DEFAULTS["repetition_penalty"]),
    max_mel_tokens: int = Form(DEFAULTS["max_mel_tokens"]),
    remove_silence: bool = Form(DEFAULTS["remove_silence"]),
    postprocess: bool = Form(DEFAULTS["postprocess"]),
    lufs: float = Form(DEFAULTS["lufs"]),
    expected_duration: float | None = Form(DEFAULTS["expected_duration"]),
    speed: float = Form(DEFAULTS["speed"]),
    pitch: float = Form(DEFAULTS["pitch"]),
):
    if runtime is None:
        raise HTTPException(status_code=503, detail="模型未初始化")
    try:
        infer_language = normalize_language(language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        temp_path = None
        emo_temp_path = None
        try:
            contents = prompt_speech.file.read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(contents)
            if emo_audio_prompt is not None:
                emo_contents = emo_audio_prompt.file.read()
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as emo_file:
                    emo_temp_path = emo_file.name
                    emo_file.write(emo_contents)
            emo_vector_list = None
            if emo_vector:
                try:
                    parsed = json.loads(emo_vector)
                    if isinstance(parsed, list) and all(
                        (isinstance(x, (int, float)) for x in parsed)
                    ):
                        emo_vector_list = [float(x) for x in parsed]
                except Exception:
                    logger.warning("emo_vector 解析失败，已忽略")
            kwargs = {
                "do_sample": bool(do_sample),
                "top_p": float(top_p),
                "temperature": float(temperature),
                "length_penalty": float(length_penalty),
                "num_beams": num_beams,
                "repetition_penalty": float(repetition_penalty),
                "max_mel_tokens": int(max_mel_tokens),
            }
            if int(top_k) > 0:
                kwargs["top_k"] = int(top_k)
            output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            tts = runtime.get()
            max_retries = 2
            tolerance = 0.05
            current_speed = float(speed)
            current_expected = None
            try:
                if expected_duration is not None:
                    current_expected = float(expected_duration)
            except Exception:
                current_expected = None
            wav = None
            sr = None
            for attempt in range(max_retries + 1):
                logger.info(
                    "开始生成语音（v2.5），文本长度: %s, language=%s, speed=%s, expected_duration=%s, attempt=%s/%s",
                    len(text),
                    infer_language,
                    current_speed,
                    current_expected,
                    attempt + 1,
                    max_retries + 1,
                )
                wav_path = tts.infer(
                    spk_audio_prompt=temp_path,
                    text=text,
                    output_path=output_path,
                    lang=infer_language,
                    emo_audio_prompt=emo_temp_path,
                    emo_alpha=float(emo_alpha),
                    emo_vector=emo_vector_list,
                    use_emo_text=bool(use_emo_text),
                    emo_text=emo_text,
                    use_random=bool(use_random),
                    interval_silence=int(interval_silence),
                    duration_factor=speed_to_duration_factor(current_speed),
                    verbose=args.verbose,
                    max_text_tokens_per_segment=int(max_text_tokens_per_sentence),
                    **kwargs,
                )
                logger.info("语音生成完成，保存到: %s", wav_path)
                wav, sr = sf.read(wav_path, dtype="float32")
                if current_expected is None or current_expected <= 0:
                    break
                try:
                    aligned_wav = trim_silence(wav, int(sr), min_silence_duration_ms=0)
                    measured = (
                        float(len(aligned_wav)) / float(sr)
                        if sr and len(aligned_wav)
                        else 0.0
                    )
                except Exception:
                    logger.exception("expected_duration: 时长测量失败，已跳过自适应")
                    break
                if measured <= 0:
                    break
                rel_err = abs(measured - current_expected) / current_expected
                if rel_err <= tolerance or attempt >= max_retries:
                    break
                new_speed = current_speed * (measured / current_expected)
                if not new_speed > 0:
                    break
                logger.info(
                    "expected_duration: measured=%.3fs expected=%.3fs rel_err=%.2f%%, update speed %.4f -> %.4f and retry",
                    measured,
                    current_expected,
                    rel_err * 100.0,
                    current_speed,
                    new_speed,
                )
                current_speed = float(new_speed)
            if wav is None or sr is None:
                raise HTTPException(
                    status_code=500, detail="语音生成失败：未获得有效音频"
                )
            if remove_silence:
                try:
                    wav = trim_silence(wav, sr)
                except Exception:
                    logger.exception("静音切除失败，已跳过")
            try:
                if float(pitch) != 0 and abs(float(pitch)) > 1e-06:
                    wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=float(pitch))
            except Exception:
                logger.exception("音高移调失败，已跳过移调")
            if postprocess:
                wav = apply_postprocess(
                    wav, sr, target_loudness=float(lufs), enable=True
                )
            buffer = BytesIO()
            sf.write(buffer, wav, sr, format="WAV")
            buffer.seek(0)
            source_name = (
                prompt_speech.filename
                or (emo_audio_prompt.filename if emo_audio_prompt is not None else None)
                or "source"
            )
            download_filename = build_download_filename(source_name, text)
            return Response(
                headers={
                    "Content-Disposition": build_content_disposition(download_filename)
                },
                content=buffer.read(),
                media_type="audio/wav",
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if emo_temp_path and os.path.exists(emo_temp_path):
                os.remove(emo_temp_path)
            if (
                "output_path" in locals()
                and output_path
                and os.path.exists(output_path)
            ):
                os.remove(output_path)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("语音生成处理错误: %s", exc)
        raise HTTPException(status_code=500, detail=f"语音生成处理错误: {str(exc)}")
