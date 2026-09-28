"""Registry dispatch must use the dedicated adapter without loading weights."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from src.model.registry import load_model, effective_config


class ModelRegistryTests(unittest.TestCase):
    def test_canvas_default_and_override_are_in_resume_identity(self):
        for name in ('djev', 'openjev'):
            config = {'models': {name: {'backend': 'diffusion'}}}
            self.assertEqual(effective_config(name, config)['canvas_length'], 64)
            config['models'][name]['canvas_length'] = 128
            self.assertEqual(effective_config(name, config)['canvas_length'], 128)

    def test_configured_models_dispatch_to_dedicated_adapters(self):
        config = json.loads((Path(__file__).resolve().parents[2] / 'configs/models.json').read_text())
        targets = {
            'jev': 'src.model.registry.Jev',
            'semif': 'src.model.semif.SemIf',
            'djev': 'src.model.djev.Djev',
            'openjev': 'src.model.openjev.OpenJev',
            'laya': 'src.model.laya.Laya',
            'open_alternative_jev': 'src.model.open_alternative_jev.LocalLibrary',
            'jeff': 'src.model.jeff.Jeff',
            'kev_0_6b': 'src.model.kev.Kev',
            'openjev_sglang': 'src.model.openjev_sglang.OpenJevSglangService',
        }
        self.assertEqual(set(targets), set(config['models']))
        for name in ('djev', 'openjev'):
            self.assertEqual(len(config['models'][name]['upstream_commit']), 40)
        for name, target in targets.items():
            with self.subTest(model=name), patch.dict(os.environ, {}, clear=True), patch(target) as constructor:
                model, entry = load_model(name, config)
                self.assertIs(model, constructor.return_value)
                constructor.assert_called_once()
                if name != 'jev':
                    self.assertEqual(constructor.call_args.args, (entry,))
                else:
                    self.assertEqual(constructor.call_args.kwargs['model'], entry['request_model'])
