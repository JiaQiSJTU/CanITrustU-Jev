"""Stream final records from a verified release directory or legacy JSONL."""
import gzip
import hashlib
import json
from pathlib import Path
from .schema import validate


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_input(path):
    """Verify every compressed file before any paid model request."""
    path = Path(path)
    manifest_path = path / 'manifest.json' if path.is_dir() else path.with_name('manifest.json')
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if path.is_dir():
        if not manifest or manifest.get('format') != 'jev-release/gzip-jsonl-v1':
            raise ValueError('Missing or unsupported release manifest')
        names = set()
        for shard in manifest['shards']:
            name = shard['path']
            if Path(name).name != name or name in names:
                raise ValueError('Invalid or duplicate shard path')
            names.add(name)
            if file_hash(path / name) != shard['sha256']:
                raise ValueError('Release shard checksum mismatch: ' + name)
        if not names or sum(s['records'] for s in manifest['shards']) != manifest['version_instances']:
            raise ValueError('Release record count mismatch')
        return manifest, manifest['stream_sha256']
    checksum = file_hash(path)
    if manifest and path.name in manifest.get('files', {}):
        if checksum != manifest['files'][path.name]['sha256']:
            raise ValueError('Input does not match frozen dataset manifest')
    return manifest, checksum


def iter_records(path, verified_manifest=None):
    """Yield records in original order, checking stream/counts on exhaustion.

    By default checks all shard hashes first. Evaluators may pass the manifest
    already returned by inspect_input to avoid repeating compressed-file I/O.
    """
    path = Path(path)
    manifest = verified_manifest
    if manifest is None:
        manifest, _ = inspect_input(path)
    digest = hashlib.sha256()
    count = 0
    shards = manifest['shards'] if path.is_dir() else [{'path': path.name}]
    for shard in shards:
        shard_count = 0
        opener = gzip.open if path.is_dir() else open
        target = path / shard['path'] if path.is_dir() else path
        with opener(target, 'rb') as stream:
            for line in stream:
                digest.update(line)
                count += 1
                shard_count += 1
                yield validate(json.loads(line))
        if 'records' in shard and shard_count != shard['records']:
            raise ValueError('Shard record count mismatch')
    if path.is_dir() and (count != manifest['version_instances'] or digest.hexdigest() != manifest['stream_sha256']):
        raise ValueError('Decompressed release checksum/count mismatch')
