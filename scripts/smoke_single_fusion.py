#!/usr/bin/env python3
"""Real synthesis through both internal entrypoints; never log reference contents."""
from pathlib import Path
import json
import hashlib
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def reference_audio():
    path = ROOT / 'examples/voice_01.wav'
    data = path.read_bytes()
    if data.startswith(b'version https://git-lfs.github.com/spec/v1'):
        oid = data.decode().split('oid sha256:', 1)[1].splitlines()[0]
        gitdir = Path(subprocess.check_output(['git', 'rev-parse', '--absolute-git-dir'], cwd=ROOT, text=True).strip())
        path = gitdir / 'lfs/objects' / oid[:2] / oid[2:4] / oid
        assert path.exists(), 'Fetch the repository reference audio LFS object before smoke'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == oid, 'Reference audio hash mismatch'
    subprocess.run(['ffprobe', '-v', 'error', str(path)], check=True)
    return path


def main():
    reference = reference_audio()
    work = Path(tempfile.mkdtemp(prefix='index-fusion-smoke-'))
    records = []
    for name, url, extra in [
        ('normal', 'http://index-api/generate', []),
        ('premium', 'http://index-api/generate', ['-H', 'X-LB-Mode: Premium']),
        ('ui-api', 'http://xique/generate', []),
    ]:
        audio = work / f'{name}.wav'
        subprocess.run(['curl', '--fail-with-body', '-sS', '--max-time', '300', '-o', str(audio),
                        '-F', 'text=这是模型服务收敛后的语音测试。',
                        '-F', f'prompt_speech=@{reference};type=audio/wav;filename=reference.wav',
                        '-F', 'seed=0', '-F', 'num_beams=1', '-F', 'do_sample=false',
                        *extra, url], check=True)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                                   'format=duration:stream=codec_name,sample_rate,channels',
                                                   '-of', 'json', str(audio)]))
        assert audio.read_bytes()[:4] == b'RIFF' and audio.stat().st_size > 1000
        assert float(probe['format']['duration']) > 0.1
        records.append({'name': name, 'bytes': audio.stat().st_size, 'probe': probe})
    with urllib.request.urlopen('http://xique/', timeout=30) as response:
        assert 'gradio' in response.read().decode().lower()
    (work / 'result.json').write_text(json.dumps(records, indent=2))
    print(f'Real Index synthesis passed: {work}', flush=True)


if __name__ == '__main__':
    main()
