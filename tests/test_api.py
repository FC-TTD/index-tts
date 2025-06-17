#!/usr/bin/env python3
import requests
import argparse
import os
import time
import soundfile as sf
import pyloudnorm as pyln
from io import BytesIO

def test_health(base_url):
    """测试健康检查端点"""
    url = f"{base_url}/health"
    print(f"测试健康检查端点: {url}")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("✅ 健康检查成功")
            print(f"响应: {response.json()}")
            return True
        else:
            print(f"❌ 健康检查失败，状态码: {response.status_code}")
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 健康检查请求异常: {str(e)}")
        return False

def test_generate_audio(base_url, text, output_file, prompt_speech, infer_mode="普通推理"):
    """测试语音生成端点"""
    url = f"{base_url}/generate"
    print(f"测试语音生成端点: {url}")
    print(f"文本: {text}")
    print(f"推理模式: {infer_mode}")
    
    # 准备表单数据
    data = {
        "text": text,
        "infer_mode": infer_mode,
        "postprocess": "true",
        "max_text_tokens_per_sentence": "120",
        "sentences_bucket_max_size": "4",
        "do_sample": "true",
        "top_p": "0.8",
        "top_k": "30",
        "temperature": "1.0",
        "length_penalty": "0.0",
        "num_beams": "3",
        "repetition_penalty": "10.0",
        "max_mel_tokens": "600"
    }
    
    files = {}
    if prompt_speech and os.path.exists(prompt_speech):
        # 正确打开文件作为二进制文件对象
        files["prompt_speech"] = (os.path.basename(prompt_speech), open(prompt_speech, "rb"), "audio/wav")
        print(f"提示语音: {prompt_speech}")
    else:
        print("错误: 提示语音文件不存在或未指定")
        return False
    
    try:
        start_time = time.time()
        response = requests.post(url, data=data, files=files)
        end_time = time.time()
        
        if response.status_code == 200:
            print(f"✅ 语音生成成功，耗时: {end_time - start_time:.2f}秒")
            
            # 保存音频文件
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"音频已保存至: {output_file}")
            
            # 获取音频信息
            audio_data = BytesIO(response.content)
            audio, sample_rate = sf.read(audio_data)
            duration = len(audio) / sample_rate
            
            # 计算平均响度
            meter = pyln.Meter(sample_rate)
            loudness = meter.integrated_loudness(audio)
            
            print(f"音频长度: {duration:.2f}秒，采样率: {sample_rate}Hz, 平均响度: {loudness:.2f} LUFS")
            
            return True
        else:
            print(f"❌ 语音生成失败，状态码: {response.status_code}")
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 语音生成请求异常: {str(e)}")
        return False
    finally:
        # 关闭文件句柄
        for file_tuple in files.values():
            if hasattr(file_tuple[1], 'close'):
                file_tuple[1].close()

def main():
    parser = argparse.ArgumentParser(description="IndexTTS API测试工具")
    parser.add_argument("--url", default="http://localhost:8000", help="API服务器URL")
    parser.add_argument("--text", default="你怎么这么傻,昨夜明明说好了你绝不会轻举妄动的", help="要合成的文本")
    parser.add_argument("--output", default="output.wav", help="输出音频文件路径")
    parser.add_argument("--prompt-speech", help="提示语音文件路径")
    parser.add_argument("--infer-mode", default="普通推理", choices=["普通推理", "批次推理"], help="推理模式")
    
    args = parser.parse_args()
    
    # 设置默认提示语音文件路径
    if not args.prompt_speech:
        args.prompt_speech = "sample_prompt.wav"
    
    # 确保提示语音文件存在
    if args.prompt_speech and not os.path.exists(args.prompt_speech):
        print(f"错误: 提示语音文件不存在: {args.prompt_speech}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 测试健康检查
    if not test_health(args.url):
        print("健康检查失败，终止测试")
        return
    
    # 测试语音生成
    test_generate_audio(
        args.url, 
        args.text, 
        args.output,
        prompt_speech=args.prompt_speech,
        infer_mode=args.infer_mode
    )

if __name__ == "__main__":
    main()
