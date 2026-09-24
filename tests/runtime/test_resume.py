import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.eval.final import kept_ok, main, read_prediction_rows
from src.model.mock import Mock
from test_reader import example


def record(version):
    row = example()
    row.update(base_id='b1', version=version, id='b1::' + version, scenario='s',
               decision_type='step_quality', actual_operator='clean')
    return row


class ResumeTests(unittest.TestCase):
    def test_partial_tail_and_errors_are_not_kept(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'predictions.jsonl'
            ok = {'base_id': 'b1', 'version': 'v0_original', 'status': 'ok', 'marker': 'first'}
            later = {'base_id': 'b1', 'version': 'v0_original', 'status': 'ok', 'marker': 'second'}
            failed = {'base_id': 'b1', 'version': 'v1_order', 'status': 'error'}
            path.write_bytes(json.dumps(ok).encode() + b'\n' + json.dumps(failed).encode() + b'\n'
                             + json.dumps(later).encode() + b'\n{"trunc')
            rows = read_prediction_rows(path)
            self.assertEqual(len(rows), 3)
            kept = kept_ok(rows)
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0]['marker'], 'second')

    def test_resume_reruns_failures_and_missing_versions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / 'versions.jsonl'
            versions = ('v0_original', 'v1_order', 'v2_label')
            data.write_text(''.join(json.dumps(record(v)) + '\n' for v in versions))
            output = root / 'out'
            argv = ['eval', '--model', 'mock', '--input', str(data), '--output', str(output)]
            with patch('sys.argv', argv):
                main()
            path = output / 'predictions.jsonl'
            saved = [json.loads(line) for line in path.read_text().splitlines()]
            rewritten = []
            for row in saved:
                if row['version'] == 'v0_original':
                    row['marker'] = 'kept'
                    rewritten.append(row)
                elif row['version'] == 'v1_order':
                    rewritten.append({'base_id': 'b1', 'version': 'v1_order', 'status': 'error', 'reason': 'HTTP 400'})
            path.write_bytes(b''.join((json.dumps(row) + '\n').encode() for row in rewritten) + b'{"partial"')
            run = json.loads((output / 'run.json').read_text())
            run['created_at'] = 'sentinel'
            (output / 'run.json').write_text(json.dumps(run))
            called = []
            real = Mock.predict

            def spy(self, item):
                called.append(item['version'])
                return real(self, item)

            with patch('sys.argv', argv + ['--resume']), patch.object(Mock, 'predict', spy):
                main()
            self.assertEqual(called, ['v1_order', 'v2_label'])
            final = [json.loads(line) for line in path.read_text().splitlines()]
            by_version = {row['version']: row for row in final}
            self.assertEqual(set(by_version), set(versions))
            self.assertEqual(by_version['v0_original']['marker'], 'kept')
            self.assertEqual(by_version['v1_order']['status'], 'ok')
            self.assertNotIn('marker', by_version['v1_order'])
            self.assertEqual(by_version['v2_label']['status'], 'ok')
            resumed = json.loads((output / 'run.json').read_text())
            self.assertEqual(resumed['created_at'], 'sentinel')
            self.assertIn('resumed_at', resumed)
            self.assertEqual(resumed['resume_kept_ok'], 1)

    def test_existing_output_still_requires_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / 'versions.jsonl'
            data.write_text(json.dumps(record('v0_original')) + '\n')
            output = root / 'out'
            output.mkdir()
            (output / 'predictions.jsonl').write_text('{}\n')
            argv = ['eval', '--model', 'mock', '--input', str(data), '--output', str(output)]
            with patch('sys.argv', argv):
                with self.assertRaises(SystemExit):
                    main()


if __name__ == '__main__':
    unittest.main()
