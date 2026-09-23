import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.dataset.reader import inspect_input, iter_records
from src.dataset.schema import request


def example():
    return dict(schema_version='jev-choice/1.0', id='fixture', dataset='fixture', group_id='g',
                state={'goal': 'Open invoice'}, instructions='Select an action.',
                options=[dict(id='z', canonical_id='open', description='Open invoice'),
                         dict(id='a', canonical_id='delete', description='Delete invoice')],
                gold='z', source={'annotation': 'private provenance'}, metadata={'private': 'private rationale'})


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.row = example()
        self.raw = (json.dumps(self.row) + '\n').encode()
        self.shard = self.root / 'part-0000.jsonl.gz'
        self.shard.write_bytes(gzip.compress(self.raw, mtime=0))
        self.manifest = dict(format='jev-release/gzip-jsonl-v1', original_samples=1,
                             version_instances=1, stream_sha256=hashlib.sha256(self.raw).hexdigest(),
                             shards=[dict(path=self.shard.name, records=1,
                                          sha256=hashlib.sha256(self.shard.read_bytes()).hexdigest())])
        self.save()

    def save(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_stream_roundtrip(self):
        self.assertEqual(list(iter_records(self.root)), [self.row])

    def test_corruption_rejected_before_iteration(self):
        self.shard.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            inspect_input(self.root)

    def test_missing_shard(self):
        self.shard.unlink()
        with self.assertRaises(FileNotFoundError):
            inspect_input(self.root)

    def test_stream_hash_checked(self):
        self.manifest['stream_sha256'] = 'incorrect'
        self.save()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            list(iter_records(self.root))

    def test_shard_count_checked(self):
        self.manifest['shards'][0]['records'] = 2
        self.manifest['version_instances'] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, 'count'):
            list(iter_records(self.root))

    def test_legacy_jsonl(self):
        path = self.root / 'versions.jsonl'
        path.write_bytes(self.raw)
        self.assertEqual(list(iter_records(path)), [self.row])

    def test_duplicate_and_unsafe_paths(self):
        self.manifest['shards'] *= 2
        self.save()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            inspect_input(self.root)
        self.manifest['shards'][0]['path'] = '../outside.gz'
        self.save()
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            inspect_input(self.root)

    def test_model_request_privacy_and_order(self):
        payload = request(self.row)
        self.assertEqual(list(payload['questions']['decision']['criteria']), ['z', 'a'])
        wire = json.dumps(payload)
        for private in ('private provenance', 'private rationale', 'canonical_id', 'gold'):
            self.assertNotIn(private, wire)


if __name__ == '__main__':
    unittest.main()
