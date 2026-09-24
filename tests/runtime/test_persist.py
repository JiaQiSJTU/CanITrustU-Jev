import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.eval.final import append_prediction, attach_provider_return
from src.model.base import CallFailure, Prediction


class PersistTests(unittest.TestCase):
    def test_each_prediction_is_fsynced_before_return(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'predictions.jsonl'
            with path.open('xb', buffering=0) as dest:
                with patch('src.eval.final.os.fsync') as sync:
                    append_prediction(dest, {'id': 'a', 'status': 'ok'})
                    self.assertEqual(path.read_bytes(), b'{"id": "a", "status": "ok"}\n')
                    self.assertEqual(sync.call_count, 1)
                    append_prediction(dest, {'id': 'b', 'status': 'error', 'reason': 'HTTP 500'})
                    self.assertEqual(sync.call_count, 2)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r['id'] for r in rows], ['a', 'b'])
            self.assertEqual(rows[1]['reason'], 'HTTP 500')

    def test_unscorable_return_is_archived_in_full(self):
        row = {'id': 'a', 'status': 'error'}
        failure = CallFailure('Choice is not a maximum-probability option',
                               response_text='{"trace":[1],"answers":{}}',
                               response={'trace': [1], 'answers': {}, 'score': float('nan')},
                               response_status=200,
                               response_headers={'X-Request-Id': 'req-9'})
        attach_provider_return(row, error=failure)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'predictions.jsonl'
            with path.open('xb', buffering=0) as dest:
                append_prediction(dest, row)
            saved = json.loads(path.read_text())
        self.assertEqual(saved['response_text'], '{"trace":[1],"answers":{}}')
        self.assertEqual(saved['raw']['trace'], [1])
        self.assertEqual(saved['raw']['score'], {'__nonfinite__': 'NaN'})
        self.assertEqual(saved['response_status'], 200)
        self.assertEqual(saved['response_headers']['X-Request-Id'], 'req-9')

    def test_scored_prediction_keeps_raw_return(self):
        row = {'id': 'a'}
        pred = Prediction('open', {'open': 1.0}, served_model='fixture', raw={'trace': {'id': 'abc'}, 'answers': {}},
                          response_text='{"trace":{"id":"abc"}}', response_status=200)
        attach_provider_return(row, pred=pred)
        self.assertEqual(row['raw']['trace']['id'], 'abc')
        self.assertEqual(row['response_text'], '{"trace":{"id":"abc"}}')
        self.assertEqual(row['response_status'], 200)


if __name__ == '__main__':
    unittest.main()
