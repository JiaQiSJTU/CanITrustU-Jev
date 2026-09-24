import copy
from io import BytesIO
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from src.model.base import CallFailure, parse_response
from src.model.jev import Jev, SystemOneOpen
from src.model.local import SemIf
from src.model.registry import load_model
from src.model.weights import Jeff, Kev, Laya
from test_reader import example as base_example


def example():
    r = base_example()
    r['options'] = [dict(id=k, canonical_id=k, description=k) for k in ['open', 'delete', 'archive']]
    r['gold'] = 'open'
    return r


class Headers:
    def __init__(self, items):
        self._items = items

    def items(self):
        return self._items


class Body:
    def __init__(self, payload, status=200, headers=None):
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self._raw = raw
        self.status = status
        self.headers = headers or Headers([('X-Request-Id', 'req-1')])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._raw


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
        with patch('src.model.jev.urlopen', return_value=Body({'model':'fixture', 'answers': {'decision': {'type':'choice', 'choice':'open', 'probabilities':{'open':1.,'delete':0.,'archive':0.}}}})) as call:
            r = example(); r['options'].reverse()
            p = Jev(key_env=None).predict(r)
            sent = json.loads(call.call_args.args[0].data)
            self.assertEqual(list(sent['questions']['decision']['criteria']), [o['id'] for o in r['options']])
            self.assertEqual(p.choice, 'open')

    def test_full_return_is_kept_when_scoring_fails(self):
        body = {'model': 'fixture', 'trace': {'logits': [0.2, 0.5, 0.3]},
                'answers': {'decision': {'type': 'choice', 'choice': 'delete',
                                         'probabilities': {'open': .8, 'delete': .1, 'archive': .1}}}}
        with patch('src.model.jev.urlopen', return_value=Body(body)):
            with self.assertRaises(CallFailure) as caught:
                Jev(key_env=None).predict(example())
        self.assertEqual(caught.exception.response['trace']['logits'], [0.2, 0.5, 0.3])
        self.assertIn('"logits"', caught.exception.response_text)
        self.assertEqual(caught.exception.response_status, 200)
        self.assertEqual(caught.exception.response_headers['X-Request-Id'], 'req-1')

    def test_http_error_body_is_kept_without_api_key(self):
        raw = b'{"detail":"rejected","echo":"SECRET"}'
        error = HTTPError('http://localhost', 400, 'Bad', Headers([('X-Request-Id', 'req-2')]), BytesIO(raw))
        with patch.dict(os.environ, {'JEV_API_KEY': 'SECRET'}):
            model = Jev()
        with patch('src.model.jev.urlopen', side_effect=error):
            with self.assertRaises(CallFailure) as caught:
                model.predict(example())
        self.assertIn('rejected', caught.exception.response_text)
        self.assertNotIn('SECRET', caught.exception.response_text)
        self.assertEqual(caught.exception.response['echo'], '[redacted]')
        self.assertEqual(caught.exception.response_status, 400)

    def test_success_keeps_fields_outside_the_score(self):
        body = {'model': 'fixture', 'usage': {'input_tokens': 3, 'output_tokens': 9}, 'trace': {'id': 'abc'},
                'answers': {'decision': {'type': 'choice', 'choice': 'open', 'probabilities': {'open': 1., 'delete': 0., 'archive': 0.}}}}
        with patch('src.model.jev.urlopen', return_value=Body(body, headers=Headers([('X-Request-Id', 'req-3')]))):
            pred = Jev(key_env=None).predict(example())
        self.assertEqual(pred.raw['trace'], {'id': 'abc'})
        self.assertEqual(pred.usage['output_tokens'], 9)
        self.assertIn('"trace"', pred.response_text)
        self.assertEqual(pred.response_headers['X-Request-Id'], 'req-3')

    def test_semif_native_alignment_without_weights(self):
        s = SemIf.__new__(SemIf); s.model=s.tokenizer=None; s.metadata={}; s.max_tokens=4096; s.model_id='fixture'
        s.score = lambda *args: {'option_ids': ['open','delete','archive'], 'probabilities':[.8,.1,.1], 'input_tokens':10}
        self.assertEqual(s.predict(example()).choice, 'open')
        returned = {'option_ids': ['delete','open','archive'], 'probabilities':[.8,.1,.1], 'input_tokens':10}
        s.score = lambda *args: returned
        with self.assertRaises(CallFailure) as caught:
            s.predict(example())
        self.assertEqual(caught.exception.response, returned)

    def test_open_weight_models_load_in_process(self):
        config = json.loads((Path(__file__).resolve().parents[2] / 'configs' / 'models.json').read_text())
        self.assertEqual(config['models']['laya']['backend'], 'laya')
        self.assertEqual(config['models']['jeff']['backend'], 'jeff')
        self.assertEqual(config['models']['kev_0_6b']['backend'], 'kev')
        self.assertNotIn('endpoint_env', config['models']['laya'])
        self.assertNotIn('endpoint_env', config['models']['jeff'])
        self.assertNotIn('command_env', config['models']['kev_0_6b'])

    def test_laya_scores_from_loaded_agent(self):
        model = Laya.__new__(Laya)
        model.model_id = 'fixture'
        seen = {}

        def predict(state, questions):
            seen['state'] = state
            seen['questions'] = questions
            return {'model': 'fixture', 'answers': {'decision': {
                'type': 'choice', 'choice': 'open',
                'probabilities': {'open': .8, 'delete': .1, 'archive': .1}}}}

        model.agent = type('Agent', (), {'predict': staticmethod(predict)})()
        record = example()
        self.assertEqual(model.predict(record).choice, 'open')
        self.assertEqual(seen['questions']['decision']['criteria']['open'], 'open')
        self.assertNotIn('gold', json.dumps(seen))

    def test_jeff_scores_from_loaded_client(self):
        model = Jeff.__new__(Jeff)
        model.model_id = 'jeff'
        model.Choice = lambda instructions, criteria: ('choice', instructions, criteria)
        model.Noul = lambda instructions: ('noul', instructions)

        class ChoiceAnswer:
            choice = 'archive'
            probabilities = {'open': .1, 'delete': .1, 'archive': .8}
            confidence = .8

        class Result:
            model = 'qwen+Jeff-1'
            choices = {'decision': ChoiceAnswer()}
            n_forward_passes = 1
            readout_modes = {'decision': 'first_token'}

        model.client = type('Client', (), {'system_one': lambda self, state, questions: Result()})()
        self.assertEqual(model.predict(example()).choice, 'archive')

    def test_kev_scores_from_loaded_checkpoint(self):
        model = Kev.__new__(Kev)
        model.model_id = 'kev-0.6b'
        model.weights = 'jaredpalmer/kev-0.6b'
        model.max_state = 8
        model.max_branch = 8
        model.tok = object()
        model.SystemOneRequest = lambda **kwargs: kwargs
        model.to_record = lambda req: ({'state': req['state']}, {'questions': 1})
        model.to_answers = lambda probs, meta: {'decision': {
            'type': 'choice', 'choice': 'delete',
            'probabilities': {'open': .2, 'delete': .6, 'archive': .2}}}
        model.output_tokens = lambda tok, answers: 0

        class Encoder:
            def encode(self, tok, rec, max_state, max_branch):
                return {'ids': [1, 2]}

            def probs(self, encoded):
                return [[.2, .6, .2]]

        model.model = Encoder()
        pred = model.predict(example())
        self.assertEqual(pred.choice, 'delete')
        self.assertEqual(pred.raw['model'], 'jaredpalmer/kev-0.6b')

    def test_registry_constructs_local_open_models(self):
        spec = {'backend': 'laya', 'weights': 'convaiinnovations/laya'}
        with patch('src.model.weights.Laya', return_value='loaded') as ctor:
            model, entry = load_model('laya', {'models': {'laya': spec}})
        self.assertEqual(model, 'loaded')
        self.assertEqual(ctor.call_args.args[0]['weights'], 'convaiinnovations/laya')
        self.assertEqual(entry['backend'], 'laya')
