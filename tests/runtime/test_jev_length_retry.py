import copy
from io import BytesIO
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from src.model.jev import Jev
from src.model.model_utils import CallFailure, estimate_tokens
from test_adapters import Body
from test_reader import example


def error(code=400, kind='max_tokens_exceeded'):
    return HTTPError('http://localhost', code, 'error', {},
                     BytesIO(json.dumps({'detail': {'error_type': kind}}).encode()))


class JevLengthRetryTests(unittest.TestCase):
    def record(self):
        r = example()
        r['state'] = 'HEAD' + 'x' * 40000 + 'TAIL'
        return r

    def response(self, record):
        labels = [o['id'] for o in record['options']]
        return {'answers': {'decision': {'type': 'choice', 'choice': labels[0],
                'probabilities': {label: float(i == 0) for i, label in enumerate(labels)}}}}

    def test_repeated_length_errors_reduce_budget_without_using_transport_retries(self):
        r = self.record()
        original = copy.deepcopy(r)
        model = Jev(key_env=None, max_input_tokens=8192, retries=0)
        with patch('src.model.jev.urlopen', side_effect=[error() for _ in range(4)] + [Body(self.response(r))]) as call:
            pred = model.predict(r)
        self.assertEqual(call.call_count, 5)
        for i, args in enumerate(call.call_args_list):
            wire = args.args[0].data.decode()
            self.assertLessEqual(estimate_tokens(wire), 8192 - 1024 * i)
            self.assertTrue(json.loads(wire)['state'].endswith('TAIL'))
        self.assertEqual(pred.truncation['max_input_tokens'], 4096)
        self.assertEqual(pred.truncation['initial_max_input_tokens'], 8192)
        self.assertEqual(pred.truncation['length_retries'], 4)
        self.assertEqual(r, original)
        self.assertEqual(model.max_input_tokens, 8192)

    def test_unshrinkable_record_stops_and_preserves_provider_error(self):
        r = self.record()
        r['state'] = ''
        with patch('src.model.jev.urlopen', side_effect=error()) as call:
            with self.assertRaises(CallFailure) as caught:
                Jev(key_env=None, max_input_tokens=30720).predict(r)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(caught.exception.response_status, 400)
        self.assertEqual(caught.exception.response['detail']['error_type'], 'max_tokens_exceeded')

    def test_positive_budget_exhaustion_stops(self):
        with patch('src.model.jev.urlopen', side_effect=[error(), error()]) as call:
            with self.assertRaises(CallFailure):
                Jev(key_env=None, max_input_tokens=2048).predict(self.record())
        self.assertLessEqual(call.call_count, 2)

    def test_other_http_errors_and_disabled_truncation_do_not_shrink(self):
        for budget, kind in ((8192, 'invalid_request'), (None, 'max_tokens_exceeded')):
            with self.subTest(budget=budget, kind=kind):
                with patch('src.model.jev.urlopen', side_effect=error(kind=kind)) as call:
                    with self.assertRaises(CallFailure):
                        Jev(key_env=None, max_input_tokens=budget).predict(self.record())
                self.assertEqual(call.call_count, 1)

    def test_transient_retry_still_uses_same_payload_and_backoff(self):
        r = self.record()
        with patch('src.model.jev.urlopen', side_effect=[error(), error(429, 'rate_limit'), Body(self.response(r))]) as call:
            with patch('src.model.jev.time.sleep') as sleep:
                pred = Jev(key_env=None, max_input_tokens=8192, retries=1).predict(r)
        self.assertEqual(call.call_args_list[1].args[0].data, call.call_args_list[2].args[0].data)
        sleep.assert_called_once_with(1)
        self.assertEqual(pred.truncation['length_retries'], 1)

    def test_transient_retries_remain_bounded(self):
        with patch('src.model.jev.urlopen', side_effect=[error(503, 'unavailable') for _ in range(3)]) as call:
            with patch('src.model.jev.time.sleep'):
                with self.assertRaises(CallFailure):
                    Jev(key_env=None, max_input_tokens=8192, retries=2).predict(self.record())
        self.assertEqual(call.call_count, 3)
