#!/usr/bin/env python3
"""Move ingress to the existing fusion before pruning redundant Swarm services."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import urllib.request
import urllib.error

import yaml

ROOT = Path(__file__).resolve().parents[1]
STACK_ID = 228
ENDPOINT_ID = 4
IMAGE = 'registry.ttd/index-tts/api:h-54be7b525694@sha256:7f3309bdacda8424cefb094cf09bee6f2b4499db1174bfa33da1b98378a3176a'


def request(path, data=None):
    base = os.environ.get('PT_URL', 'http://ttd-cctv:9000').rstrip('/')
    key = os.environ['PT_API_KEY']
    req = urllib.request.Request(base + path, headers={'X-API-Key': key, 'Content-Type': 'application/json'},
                                 data=None if data is None else json.dumps(data).encode(),
                                 method='GET' if data is None else 'PUT')
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def stack_text():
    return request(f'/api/stacks/{STACK_ID}/file')['StackFileContent']


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def env_map(service):
    env = service.get('environment', {})
    return env if isinstance(env, dict) else dict(value.split('=', 1) for value in env)


def compare_fusion(current, desired):
    """Only ingress labels and already-resolved image notation may change."""
    left, right = copy.deepcopy(current), copy.deepcopy(desired)
    assert left['image'] in (IMAGE, IMAGE.split('@')[0]), 'Unexpected current image; re-audit before release'
    assert right['image'] == IMAGE
    for value in (left, right):
        value['image'] = IMAGE
        value['environment'] = env_map(value)
        value['deploy'].pop('labels', None)
    assert left == right, 'Fusion runtime definition differs beyond the approved label change'


def build_transition(current, desired):
    assert set(current['services']) == {'api-1', 'api-2', 'api-3'}
    assert set(desired['services']) == {'api-3'}
    compare_fusion(current['services']['api-3'], desired['services']['api-3'])
    transition = copy.deepcopy(current)
    transition['services']['api-3'] = copy.deepcopy(desired['services']['api-3'])
    for name in ('api-1', 'api-2'):
        service = transition['services'][name]
        assert service['image'] in (IMAGE, IMAGE.split('@')[0]), f'Unexpected {name} image'
        service['image'] = IMAGE
        service['deploy']['labels'] = {key: value for key, value in service['deploy'].get('labels', {}).items()
                                        if not key.startswith('caddy')}
    return transition


def index_upstreams(config):
    found = []
    def walk(routes, hosts=()):
        for route in routes:
            for match in route.get('match') or [{}]:
                selected = match.get('host', hosts)
                for handler in route.get('handle', []):
                    if handler.get('handler') == 'subroute':
                        walk(handler.get('routes', []), selected)
                    if handler.get('handler') == 'reverse_proxy' and 'index-api' in selected:
                        found.extend(item['dial'] for item in handler.get('upstreams', []))
    for server in config['apps']['http']['servers'].values():
        walk(server.get('routes', []))
    return found


def wait_route():
    for _ in range(30):
        with urllib.request.urlopen('http://ttd-server/caddy_api/config/', timeout=15) as response:
            upstreams = index_upstreams(json.load(response))
        if upstreams and set(upstreams) == {'index-tts_api-3.:8000'}:
            try:
                for url in ('http://index-api/health', 'http://xique/health'):
                    with urllib.request.urlopen(url, timeout=15) as health:
                        assert health.status == 200
                return
            except (urllib.error.URLError, TimeoutError):
                pass
        time.sleep(2)
    raise RuntimeError('index-api has not converged to api-3; old services retained')


def wait_drained():
    # Health executes on the same event loop as inference. Then require no open
    # inbound HTTP connections, including responses which are still streaming.
    probe = """import urllib.request
urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=60).read()
from pathlib import Path
count=0
for p in ('/proc/net/tcp','/proc/net/tcp6'):
 for line in Path(p).read_text().splitlines()[1:]:
  fields=line.split()
  count += fields[1].split(':')[1]=='1F40' and fields[3]=='01'
print(count)
"""
    for name in ('api-1', 'api-2'):
        command = ['docker', 'ps', '-q', '--filter', f'label=com.docker.swarm.service.name=index-tts_{name}']
        container = subprocess.check_output(['ssh', 'root@ttd-worker', shlex.join(command)], text=True).strip()
        assert container and '\n' not in container, f'Unexpected task count for {name}'
        for _ in range(20):
            result = subprocess.run(['ssh', 'root@ttd-worker', shlex.join(['docker', 'exec', container, 'python', '-c', probe])],
                                    text=True, capture_output=True, timeout=70)
            if result.returncode == 0 and result.stdout.strip() == '0':
                break
            time.sleep(3)
        else:
            raise RuntimeError(f'{name} still has HTTP connections; old services retained')


def update(text, env, expected):
    metadata = request(f'/api/stacks/{STACK_ID}')
    assert metadata['Name'] == 'index-tts' and metadata['EndpointId'] == ENDPOINT_ID
    assert sorted(metadata.get('Env', []), key=lambda item: item['name']) == sorted(env, key=lambda item: item['name']), 'Portainer environment changed concurrently; refusing overwrite'
    assert digest(stack_text()) == expected, 'Portainer stack changed concurrently; refusing overwrite'
    request(f'/api/stacks/{STACK_ID}?endpointId={ENDPOINT_ID}',
            {'stackFileContent': text, 'prune': True, 'pullImage': False, 'env': env})
    actual = stack_text()
    assert yaml.safe_load(actual) == yaml.safe_load(text), 'Portainer management state did not converge'
    return digest(actual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--smoke-command', required=True, help='Executable business smoke; failure retains old services')
    args = parser.parse_args()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'docker/docker-stack.yml', 'scripts/consolidate_swarm.py', 'scripts/smoke_single_fusion.py', 'deploy.sh'], cwd=ROOT, text=True)
    assert not dirty, 'Commit deployment inputs before release'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    desired_text = subprocess.check_output(['git', 'show', f'{commit}:docker/docker-stack.yml'], cwd=ROOT, text=True)
    desired = yaml.safe_load(desired_text)
    metadata = request(f'/api/stacks/{STACK_ID}')
    assert metadata['Name'] == 'index-tts' and metadata['EndpointId'] == ENDPOINT_ID
    before_text = stack_text()
    before = yaml.safe_load(before_text)
    transition = build_transition(before, desired)
    work = Path(tempfile.mkdtemp(prefix='index-consolidation-'))
    work.chmod(0o700)
    for name, content in [('rollback.yml', before_text), ('transition.yml', yaml.safe_dump(transition, sort_keys=False)), ('desired.yml', desired_text)]:
        (work / name).write_text(content)
        (work / name).chmod(0o600)
    rollback = {'stack_id': STACK_ID, 'endpoint_id': ENDPOINT_ID,
                'body': {'stackFileContent': before_text, 'env': metadata.get('Env', []), 'prune': True, 'pullImage': False}}
    (work / 'rollback.json').write_text(json.dumps(rollback, indent=2))
    (work / 'rollback.json').chmod(0o600)
    record = {'source_commit': commit, 'image': IMAGE, 'stack_id': STACK_ID, 'endpoint_id': ENDPOINT_ID, 'desired_sha256': digest(desired_text), 'rollback_sha256': digest(before_text)}
    def stage(value):
        record['stage'] = value
        (work / 'release.json').write_text(json.dumps(record, indent=2))
    stage('prepared')
    print(f'Prepared {commit}; rollback and evidence: {work}', flush=True)
    if not args.apply:
        return
    from smoke_single_fusion import reference_audio
    reference_audio()
    try:
        stage('switching_ingress')
        current = update((work / 'transition.yml').read_text(), metadata.get('Env', []), digest(before_text))
        wait_route()
        stage('verifying_fusion')
        subprocess.run(shlex.split(args.smoke_command), check=True)
        stage('draining_old_tasks')
        wait_drained()
        stage('pruning_old_services')
        current = update(desired_text, metadata.get('Env', []), current)
        wait_route()
        stage('verifying_final_state')
        subprocess.run(shlex.split(args.smoke_command), check=True)
        record.update({'management_sha256': current, 'status': 'success'})
        stage('complete')
    except Exception as error:
        record.update({'status': 'failed', 'failure_type': type(error).__name__})
        stage(record['stage'])
        raise
    print(f'Index consolidated to api-3; evidence: {work}', flush=True)


if __name__ == '__main__':
    main()
