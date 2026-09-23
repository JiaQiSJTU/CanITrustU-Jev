import copy
import json
import unittest
from unittest.mock import patch
from src.model.base import parse_response
from src.model.jev import Jev, SystemOneOpen
from src.model.local import SemIf
from test_reader import example as base_example


def example():
    r = base_example()
    r['options'] = [dict(id=k, canonical_id=k, description=k) for k in ['open', 'delete', 'archive']]
    r['gold'] = 'open'
    return r


class AdapterTests(unittest.TestCase):
    def test_strict_response_checks(self):
        opts = example()['options']
        payload = {'answers': {'decision': {'type': 'choice', 'choice': 'open', 'probabilities': {'open': .8, 'delete': .1, 'archive': .1}}}}
        self.assertEqual(parse_response(payload, opts).choice, 'open')
        bad = copy.deepcopy(payload); bad['answers']['decision']['probabilities']['delete'] = float('nan')
        with self.assertRaises(ValueError): parse_response(bad, opts)
        bad = copy.deepcopy(payload); bad['answers']['decision']['choice'] = 'delete'
        with self.assertRaises(ValueError): parse_response(bad, opts)
        bad = copy.deepcopy(payload); del bad['answers']['decision']['probabilities']['archive']
        with self.assertRaises(ValueError): parse_response(bad, opts)

    def test_system_one_protocol_translation(self):
        backend = SystemOneOpen(endpoint='http://localhost/decide', key_env=None)
        payload = backend.make_payload(example())
        self.assertIsInstance(payload['questions'], list)
        self.assertIn('options', payload['questions'][0])
        self.assertNotIn('criteria', payload['questions'][0])
        raw = {'answers': [{'id': 'decision', 'type': 'choice', 'choice': 'open', 'probabilities': {'open': 1., 'delete': 0., 'archive': 0.}}]}
        self.assertEqual(parse_response(backend.normalize_response(raw), example()['options']).choice, 'open')

    def test_http_wire_uses_actual_option_order(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return json.dumps({'model':'fixture', 'answers': {'decision': {'type':'choice', 'choice':'open', 'probabilities':{'open':1.,'delete':0.,'archive':0.}}}}).encode()
        with patch('src.model.jev.urlopen', return_value=Response()) as call:
            r = example(); r['options'].reverse()
            p = Jev(key_env=None).predict(r)
            sent = json.loads(call.call_args.args[0].data)
            self.assertEqual(list(sent['questions']['decision']['criteria']), [o['id'] for o in r['options']])
            self.assertEqual(p.choice, 'open')

    def test_semif_native_alignment_without_weights(self):
        s = SemIf.__new__(SemIf); s.model=s.tokenizer=None; s.metadata={}; s.max_tokens=4096; s.model_id='fixture'
        s.score = lambda *args: {'option_ids': ['open','delete','archive'], 'probabilities':[.8,.1,.1], 'input_tokens':10}
        self.assertEqual(s.predict(example()).choice, 'open')
        s.score = lambda *args: {'option_ids': ['delete','open','archive'], 'probabilities':[.8,.1,.1], 'input_tokens':10}
        with self.assertRaises(ValueError): s.predict(example())
