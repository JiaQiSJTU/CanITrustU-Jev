import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from src.model.model_utils import local_context, check_input_tokens
from src.model.jeff import Jeff
from src.model.semif import SemIf
from src.model.open_alternative_jev import LocalLibrary
from src.model.openjev_sglang import OpenJevSglang
from src.model.diffusion import DiffusionLocal
from src.model.mock import Mock
from src.eval.final import main
from test_resume import record


class LocalContextTests(unittest.TestCase):
    def test_nested_config_and_method_reserves(self):
        cfg = NS(max_position_embeddings=999, text_config=NS(max_position_embeddings=262144))
        for reserve in (0, 1, 256):
            c = local_context(cfg, reserve=reserve)
            self.assertEqual(c['max_input_tokens'], 262144 - reserve)
            check_input_tokens(262144 - reserve, c)
            with self.assertRaises(ValueError):
                check_input_tokens(262145 - reserve, c)
        self.assertEqual(local_context({'n_positions': 2048})['max_input_tokens'], 2048)

    def test_invalid_or_unknown_limits_do_not_guess(self):
        for cfg in ({}, {'max_position_embeddings': True}, {'max_position_embeddings': 10**30}):
            with self.assertRaises(ValueError):
                local_context(cfg)
        for requested in (0, -1, True, 16.0, 17):
            with self.assertRaises(ValueError):
                local_context({'max_position_embeddings': 16}, requested=requested)
        with self.assertRaises(ValueError):
            local_context({'max_position_embeddings': 16}, reserve=16)

    def test_semif_uses_loaded_text_limit(self):
        core = NS(load_causal_model=lambda *a: (NS(config=NS(text_config=NS(max_position_embeddings=262144))), None, {}))
        with patch.dict('sys.modules', {'semif_phase1': NS(), 'semif_phase1.core': core,
                                       'semif_phase1.direct': NS(score=lambda *a: None)}):
            model = SemIf({'request_model': 'fixture', 'weights_revision': None})
        self.assertEqual(model.max_tokens, 262144)

    def test_so1_uses_loaded_backbone(self):
        engine = NS(backend=NS(model=NS(config=NS(max_position_embeddings=262144))))
        module = NS(Choice=object, Decider=NS(from_pretrained=lambda *a, **kw: engine))
        with patch.dict('sys.modules', {'so1': module}):
            adapter = LocalLibrary({'weights': 'fixture'})
        self.assertEqual(adapter.context['max_input_tokens'], 262144)

    def test_jeff_reserves_sequence_labels_only(self):
        model = Jeff.__new__(Jeff)
        model.context = local_context({'max_position_embeddings': 32})
        model.client = NS(tokenizer=None)
        model.label_space = lambda question: ['a', 'b']
        model.readout = NS(label_token_variants=lambda *a: {'a': [1], 'b': [1, 2, 3]},
                           choose_mode=lambda variants: 'sequence')
        self.assertEqual(model._question_context(None)['max_input_tokens'], 29)
        model.readout.choose_mode = lambda variants: 'first_token'
        self.assertEqual(model._question_context(None)['max_input_tokens'], 32)
        self.assertEqual(model.context['max_input_tokens'], 32)

    def test_jeff_replaces_native_default(self):
        client = NS(model=NS(config=NS(max_position_embeddings=262144)), max_length=2048)
        package = NS(readout=NS(), schema=NS(label_space=lambda q: []))
        module = NS(Choice=object, Noul=object, SystemOneClient=lambda **kw: client)
        with patch.dict('sys.modules', {'jev_clf': package, 'jev_clf.client': module}):
            model = Jeff({})
        self.assertEqual(model.client.max_length, 262144)

    def test_diffusion_and_sglang_constructor_limits(self):
        cfg = NS(text_config=NS(max_position_embeddings=262144, vocab_size=100), canvas_length=256)
        tokenizer = NS(from_pretrained=lambda *a: NS())
        native = NS(config=cfg, eval=lambda: None)
        loaded_configs = []
        def load_diffusion(*args, **kwargs):
            loaded_configs.append(kwargs['config'].canvas_length)
            return native
        transformers = NS(AutoTokenizer=tokenizer, AutoConfig=NS(from_pretrained=lambda *a: cfg),
                          DiffusionGemmaForBlockDiffusion=NS(from_pretrained=load_diffusion))
        with patch.dict('sys.modules', {'transformers': transformers}), patch('src.model.diffusion.scaffold_ids', return_value=[]):
            model = DiffusionLocal({'weights': 'fixture'})
        self.assertEqual(model.canvas_length, 64)
        self.assertEqual(model.model.config.canvas_length, 64)
        self.assertEqual(loaded_configs, [64])
        self.assertEqual(model.context['max_input_tokens'], 262080)
        calls = []
        def make_engine(**kw):
            calls.append(kw)
            return NS(get_server_info=lambda: {'max_req_input_len': 262139})
        engine = NS(Engine=make_engine)
        with patch.dict('sys.modules', {'transformers': transformers, 'sglang': engine}), patch('src.model.openjev_sglang.single_token_labels', return_value=[]):
            model = OpenJevSglang({'weights': 'fixture'})
        self.assertEqual(model.context['max_input_tokens'], 262138)
        self.assertEqual(calls[0]['context_length'], 262144)

    def test_resume_rejects_changed_resolved_limit_before_rewrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / 'data.jsonl'
            data.write_text(json.dumps(record('v0_original')) + '\n')
            output = root / 'out'
            args = ['eval', '--model', 'mock', '--input', str(data), '--output', str(output)]
            with patch('sys.argv', args):
                main()
            before = (output / 'predictions.jsonl').read_bytes()
            with patch.object(Mock, 'context', {'max_input_tokens': 123}, create=True), patch('sys.argv', args + ['--resume']):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual((output / 'predictions.jsonl').read_bytes(), before)
