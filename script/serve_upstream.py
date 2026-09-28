"""Build and run pinned upstream backends on loopback, without rewriting their APIs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.model_utils import local_context

SOURCES = json.loads((ROOT / 'configs/upstream_sources.json').read_text())
CACHE = ROOT / '.deps/native-upstream'
PORTS = {'djev': 8011, 'openjev': 8012, 'openjev_sglang': 8013}
WEIGHTS = {'djev': 'DJEV_WEIGHTS', 'openjev': 'OPENJEV_WEIGHTS', 'openjev_sglang': 'OPENJEV_SGLANG_WEIGHTS'}
LOCAL_WEIGHTS = ROOT / 'configs/local_backend_weights.json'


def image_name(name):
    return 'canitrustu/' + name.replace('_', '-') + ':' + SOURCES[name]['commit'][:12]


def verify_source(name):
    path = CACHE / name
    spec = SOURCES[name]
    if not (path / '.upstream-commit').exists() or (path / '.upstream-commit').read_text().strip() != spec['commit']:
        raise ValueError(f'{name}: missing pinned source; run with --build')
    for filename, expected in spec['source_sha256'].items():
        if hashlib.sha256((path / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'{name}: modified upstream source: {filename}')


def prepare_source(name):
    target = CACHE / name
    if target.exists():
        verify_source(name)
        return
    spec = SOURCES[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CACHE) as tmp:
        archive = Path(tmp) / 'source.tar.gz'
        subprocess.run(['curl', '-fL', '--retry', '3',
                        f"https://codeload.github.com/{spec['repository']}/tar.gz/{spec['commit']}",
                        '-o', str(archive)], check=True)
        staging = Path(tmp) / 'source'
        staging.mkdir()
        with tarfile.open(archive) as tar:
            if tar.pax_headers.get('comment') != spec['commit']:
                raise ValueError('Downloaded archive does not identify the pinned commit')
            for member in tar:
                if not member.isfile():
                    continue
                relative = Path(*Path(member.name).parts[1:])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe upstream archive path')
                dest = staging / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar.extractfile(member).read())
                dest.chmod(member.mode & 0o777)
        (staging / '.upstream-commit').write_text(spec['commit'] + '\n')
        staging.rename(target)
    verify_source(name)


def build_commands(name):
    if name == 'openjev_sglang':
        return [['docker', 'build', '-f', str(ROOT / 'script/backend/Dockerfile.sglang'),
                 '-t', image_name(name), str(CACHE)]]
    source = str(CACHE / 'openjev')
    base = 'canitrustu/openjev-base:' + SOURCES['openjev']['commit'][:12]
    commands = [
        ['docker', 'build', '-f', source + '/docker/Dockerfile.base', '-t', base, source],
        ['docker', 'build', '-f', source + '/docker/Dockerfile', '--build-arg', 'BASE_IMAGE=' + base,
         '-t', image_name('openjev'), source],
    ]
    if name == 'djev':
        commands.append(['docker', 'build', '-f', str(ROOT / 'script/backend/Dockerfile.djev'),
                         '--build-arg', 'BASE_IMAGE=' + image_name('openjev'),
                         '-t', image_name(name), str(CACHE)])
    return commands


def run_command(name, weights, port, gpus):
    cfg = json.loads((weights / 'config.json').read_text())
    limit = local_context(cfg)['backbone_max_tokens']
    env = {'MODEL_MAX_TOKENS': str(limit)}
    if name == 'openjev':
        env.update(OPENJEV_MODEL='/weights', OPENJEV_TOKENIZER='/weights',
                   OPENJEV_MAX_MODEL_LEN=str(limit))
    elif name == 'openjev_sglang':
        # One evaluation question: prefix warmup plus the full branch may submit
        # up to twice the maximum context. Only token budgets differ from upstream.
        env.update(OPENJEV_MAX_INPUT_TOKENS=str(limit), OPENJEV_MAX_TOTAL_INPUT_TOKENS=str(2 * limit))
    command = ['docker', 'run', '--rm', '--gpus', gpus, '--ipc=host',
               '--name', 'canitrustu-' + name.replace('_', '-'),
               '-p', f'127.0.0.1:{port}:8080',
               '--mount', f'type=bind,src={weights},dst=/weights,readonly']
    for key, value in env.items():
        command.extend(['-e', f'{key}={value}'])
    command.append(image_name(name))
    return command, limit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=PORTS)
    parser.add_argument('--build', action='store_true', help='Fetch verified upstream sources and build images; do not start GPUs')
    parser.add_argument('--dry-run', action='store_true', help='Print exact Docker commands without downloads/build/start')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--port', type=int)
    parser.add_argument('--gpus', default='all', help='Docker GPU selector, e.g. device=0')
    args = parser.parse_args()
    name = args.model
    if args.build:
        commands = build_commands(name)
        if not args.dry_run:
            for source in ({'openjev', 'djev'} if name == 'djev' else {name}):
                prepare_source(source)
            shutil.copyfile(ROOT / 'script/backend/djev-entrypoint.sh', CACHE / 'djev-entrypoint.sh')
        for command in commands:
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
        return
    key = WEIGHTS[name]
    local_weights = json.loads(LOCAL_WEIGHTS.read_text()) if LOCAL_WEIGHTS.is_file() else {}
    selected_weights = args.weights or os.environ.get(key) or local_weights.get(name)
    if not selected_weights:
        parser.error(f'Set --weights or {key} to a local checkpoint directory')
    weights = Path(selected_weights).resolve()
    if not (weights / 'config.json').is_file():
        parser.error(f'Missing local checkpoint config: {weights}/config.json; set --weights or {key}')
    port = PORTS[name] if args.port is None else args.port
    if not 1 <= port <= 65535:
        parser.error('Port must be between 1 and 65535')
    command, limit = run_command(name, weights, port, args.gpus)
    print(shlex.join(command), flush=True)
    print(f'Evaluation endpoint: http://127.0.0.1:{port}/v1/systemone; backbone max tokens: {limit}', flush=True)
    if args.dry_run:
        return
    try:
        image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image_name(name)], text=True).strip()
    except subprocess.CalledProcessError:
        parser.error(f'Build the backend first: bash script/serve_{name}.sh --build')
    manifest = ROOT / 'results/backend-launches' / (name + '.json')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'model': name, 'upstream': SOURCES[name], 'image_id': image_id,
        'weights': str(weights), 'model_config_sha256': hashlib.sha256((weights / 'config.json').read_bytes()).hexdigest(),
        'backbone_max_tokens': limit, 'command': command,
        'endpoint': f'http://127.0.0.1:{port}/v1/systemone',
        'note': 'Launch specification, not proof of readiness; wait for upstream startup/health.'}, indent=2) + '\n')
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
