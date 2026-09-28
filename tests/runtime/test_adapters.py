import copy
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from src.model.model_utils import CallFailure, parse_response
from src.model.jev import Jev
from src.model.system_one_open import SystemOneOpen
from src.model.semif import SemIf
from src.model.diffusion import DiffusionLocal, slot_distribution
from src.model.openjev_sglang import OpenJevSglang
from src.model.registry import load_model, effective_config
from src.model.jeff import Jeff
from src.model.kev import Kev, pin_local_base, encode_kev_with_budget
from src.model.laya import Laya
from src.model.model_utils import prefer_env, resolve_hub_snapshot
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
    def test_kev_context_uses_backbone_limit(self):
        adapter = Kev.__new__(Kev)
        adapter.model = SimpleNamespace(lm=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=32768)))
        adapter._configure_context()
        self.assertEqual(adapter.max_state, 32768)
        self.assertEqual(adapter.max_branch, 32768)
        with self.assertRaisesRegex(ValueError, 'exceeds available limit'):
            adapter._configure_context(32769)

    def test_kev_reserves_branch_space_before_truncating_state(self):
        class Encoder:
            def encode(self, tok, rec, max_state, max_branch):
                state_len = min(len(rec['state']) + 1, max_state)
                if state_len + 3 > max_branch:
                    raise ValueError('branch too long')
                return {'ids': [0] * (state_len + 3),
                        'seg': [0] * state_len + [1] * 3,
                        'state_truncated': len(rec['state']) + 1 > max_state}
        enc, context = encode_kev_with_budget(Encoder(), None, {'state': 'x' * 20}, 8, 8)
        self.assertEqual(len(enc['ids']), 8)
        self.assertEqual(context['state_budget'], 5)
        self.assertTrue(context['state_truncated'])
        enc, context = encode_kev_with_budget(Encoder(), None, {'state': 'x'}, 8, 8)
        self.assertEqual(len(enc['ids']), 5)
        self.assertFalse(context['state_truncated'])
        enc, context = encode_kev_with_budget(Encoder(), None, {'state': 'x' * 40000}, 32768, 32768)
        self.assertEqual(len(enc['ids']), 32768)
        self.assertEqual(context['state_budget'], 32765)
        self.assertTrue(context['state_truncated'])

    def test_laya_records_gpu_to_cpu_fallback(self):
        adapter = Laya.__new__(Laya)
        adapter.model_id = 'fixture'
        parameter = SimpleNamespace(device='cuda:0')
        agent = SimpleNamespace(device='cuda', cfg={'max_len': 8192},
                                model=SimpleNamespace(parameters=lambda: iter([parameter])))
        def predict(state, questions):
            agent.device = 'cpu'
            parameter.device = 'cpu'
            return {'answers': {'decision': {'type': 'choice', 'choice': 'open',
                    'probabilities': {'open': 1., 'delete': 0., 'archive': 0.}}}}
        agent.predict = predict
        adapter.agent = agent
        pred = adapter.predict(example())
        meta = pred.raw['backend_metadata']
        self.assertEqual(meta['device_before']['parameter_devices'], ['cuda:0'])
        self.assertEqual(meta['device_after']['parameter_devices'], ['cpu'])
        self.assertEqual(meta['max_len'], 8192)

    def test_laya_context_budget_and_encoder_limit(self):
        agent = SimpleNamespace(
            cfg={'max_len': 1024, 'head_max_len': 256},
            model=SimpleNamespace(encoder=SimpleNamespace(
                config=SimpleNamespace(max_position_embeddings=8192))))
        module = SimpleNamespace(load=lambda *args, **kwargs: agent)
        with patch.dict('sys.modules', {'laya': module}):
            Laya({})
            self.assertEqual(agent.cfg, {'max_len': 8192, 'head_max_len': 256})
            with self.assertRaisesRegex(ValueError, 'exceeds available limit'):
                Laya({'max_len': 8193})
            for value in (0, -1, True, 8192.0):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    Laya({'max_len': value})

    def test_effective_identity_tracks_weights_but_not_api_key(self):
        config = json.loads((Path(__file__).resolve().parents[2] / 'configs' / 'models.json').read_text())
        with patch.dict(os.environ, {'SEMIF_MODEL': '/tmp/base-a', 'SEMIF_REVISION': ''}):
            first = effective_config('semif', config)
        with patch.dict(os.environ, {'SEMIF_MODEL': '/tmp/base-b', 'SEMIF_REVISION': ''}):
            second = effective_config('semif', config)
        self.assertNotEqual(first, second)
        self.assertEqual(first['request_model'], '/tmp/base-a')
        self.assertIsNone(first['weights_revision'])
        with patch.dict(os.environ, {'JEV_API_KEY': 'SECRET-ONE'}):
            first = effective_config('jev', config)
        with patch.dict(os.environ, {'JEV_API_KEY': 'SECRET-TWO'}):
            self.assertEqual(first, effective_config('jev', config))
        self.assertNotIn('SECRET', json.dumps(first))

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

    def test_weight_env_overrides_config_path(self):
        with patch.dict(os.environ, {'LAYA_WEIGHTS': '/tmp/laya'}):
            self.assertEqual(prefer_env('LAYA_WEIGHTS', 'convaiinnovations/laya'), '/tmp/laya')
        with patch.dict(os.environ, {'LAYA_WEIGHTS': ''}):
            self.assertEqual(prefer_env('LAYA_WEIGHTS', 'convaiinnovations/laya'), 'convaiinnovations/laya')

    def test_open_weight_models_use_native_library_or_upstream_service(self):
        config = json.loads((Path(__file__).resolve().parents[2] / 'configs' / 'models.json').read_text())
        self.assertEqual(config['models']['laya']['backend'], 'laya')
        self.assertEqual(config['models']['jeff']['backend'], 'jeff')
        self.assertEqual(config['models']['kev_0_6b']['backend'], 'kev')
        self.assertEqual(config['models']['djev']['backend'], 'djev_upstream')
        self.assertEqual(config['models']['openjev']['backend'], 'openjev_upstream')
        self.assertEqual(config['models']['openjev_sglang']['backend'], 'sglang_upstream')
        self.assertNotIn('endpoint_env', config['models']['laya'])
        self.assertNotIn('endpoint_env', config['models']['jeff'])
        self.assertIn('endpoint_env', config['models']['djev'])
        self.assertIn('endpoint_env', config['models']['openjev'])
        self.assertIn('endpoint_env', config['models']['openjev_sglang'])
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
        model._question_context = lambda question: {'max_input_tokens': 1000}
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
                return {'ids': [1, 2], 'seg': [0, 1], 'state_truncated': False}

            def probs(self, encoded):
                return [[.2, .6, .2]]

        model.model = Encoder()
        pred = model.predict(example())
        self.assertEqual(pred.choice, 'delete')
        self.assertEqual(pred.raw['model'], 'jaredpalmer/kev-0.6b')

    def test_kev_base_uses_local_snapshot(self):
        meta = type('Meta', (), {'base': 'Qwen/Qwen3-0.6B-Base', 'base_revision': 'da87bfb'})()
        pin_local_base(meta, '/tmp/qwen-base')
        self.assertEqual(meta.base, '/tmp/qwen-base')
        self.assertIsNone(meta.base_revision)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / 'snapshots' / 'abc'
            snapshot.mkdir(parents=True)
            (snapshot / 'config.json').write_text('{}', encoding='utf-8')
            refs = root / 'refs'
            refs.mkdir()
            (refs / 'main').write_text('abc\n', encoding='utf-8')
            self.assertEqual(resolve_hub_snapshot(str(root / 'snapshots')), str(snapshot))
            self.assertEqual(resolve_hub_snapshot(str(snapshot)), str(snapshot))

    def test_diffusion_read_scores_the_seeded_slot(self):
        model = DiffusionLocal.__new__(DiffusionLocal)
        model.model_id = 'djev'
        model.weights = '/tmp/diffusiongemma'
        model.upstream_method = "djev"
        model.choice_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        model.canvas_length = 64
        model.canvas_step = 16
        model.confidence_method = "max_probability"
        model.input_limit = None
        model.vocab = 400
        model.scaffold = [10, 11]
        model.context = {'max_input_tokens': 1000, 'backbone_max_tokens': 1064}

        class Tok:
            def encode(self, text, add_special_tokens=False):
                if text == '<|channel>thought\n':
                    return [10]
                if text == '<channel|>':
                    return [11]
                if text == '<|channel>thought\n<channel|>':
                    return [10, 11]
                if text.startswith('decision: '):
                    return [7, 300 + ord(text[-1])]
                raise AssertionError(text)

            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=False):
                return [1, 2]

        model.tok = Tok()
        seen = {}

        def slot_logprobs(prompt_ids, canvas_ids, pos, label_ids):
            seen['pos'] = pos
            seen['canvas'] = list(canvas_ids)
            return {token_id: (0.0 if token_id == label_ids[1] else -20.0) for token_id in label_ids}

        model.slot_logprobs = slot_logprobs
        pred = model.predict(example())
        self.assertEqual(pred.choice, 'delete')
        self.assertIn('pos', seen)
        self.assertEqual(len(seen['canvas']), 16)
        self.assertEqual(pred.raw['backend_metadata']['canvas_length'], 16)
        probs, entropy = slot_distribution({1: 0.0, 2: -20.0}, [1, 2])
        self.assertAlmostEqual(probs[0], 1.0, places=5)
        self.assertLess(entropy, 0.1)

    def test_sglang_read_scores_selected_label_tokens(self):
        model = OpenJevSglang.__new__(OpenJevSglang)
        model.model_id = 'jev-latest'
        model.weights = '/tmp/qwen'
        model.labels = [(chr(65 + i), 10 + i) for i in range(64)]
        model.context = {'max_input_tokens': 1000, 'backbone_max_tokens': 1064}

        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [len(text)]

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
                return messages[-1]['content'] + '<gen>'

        model.tok = Tok()

        def generate(input_ids, label_ids=None):
            if label_ids is None:
                return [], len(input_ids)
            rows = [(0.0 if token_id == label_ids[2] else -8.0, token_id) for token_id in label_ids]
            return [row[0] for row in rows], len(input_ids)

        model.generate = generate
        self.assertEqual(model.predict(example()).choice, 'archive')

    def test_registry_constructs_local_open_models(self):
        spec = {'backend': 'laya', 'weights': 'convaiinnovations/laya'}
        with patch('src.model.laya.Laya', return_value='loaded') as ctor:
            model, entry = load_model('laya', {'models': {'laya': spec}})
        self.assertEqual(model, 'loaded')
        self.assertEqual(ctor.call_args.args[0]['weights'], 'convaiinnovations/laya')
        self.assertEqual(entry['backend'], 'laya')
