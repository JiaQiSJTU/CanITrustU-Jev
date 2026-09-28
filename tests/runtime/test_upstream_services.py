"""Native services own inference; clients preserve requests and returned scores."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.model.registry import load_model
from test_adapters import Body, example

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('serve_upstream', ROOT / 'script/serve_upstream.py')
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


class UpstreamServiceTests(unittest.TestCase):
    def test_native_transport_keeps_confidence_and_question_ids(self):
        config = json.loads((ROOT / 'configs/models.json').read_text())
        payload = {'answers': {'decision': {'type': 'choice', 'choice': 'open',
                   'probabilities': {'open': .8, 'delete': .1, 'archive': .1},
                   'confidence': .37}}, 'usage': {'input_tokens': 42}}
        for name in serve.PORTS:
            with self.subTest(name=name), patch.dict('os.environ', {}, clear=True):
                model, entry = load_model(name, config)
                with patch('src.model.jev.urlopen', return_value=Body(payload)) as call:
                    prediction = model.predict(example())
                sent = json.loads(call.call_args.args[0].data)
                self.assertEqual(call.call_args.args[0].full_url, entry['default_endpoint'])
                self.assertEqual(sent['model'], entry['request_model'])
                self.assertEqual(list(sent['questions']), ['decision'])
                self.assertEqual(sent['questions']['decision']['criteria'],
                                 {'open': 'open', 'delete': 'delete', 'archive': 'archive'})
                self.assertNotIn('seed', sent)
                self.assertEqual(prediction.confidence, .37)
                self.assertEqual(prediction.raw, payload)

    def test_launcher_mounts_weights_and_changes_only_token_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp)
            (weights / 'config.json').write_text(json.dumps({'text_config': {'max_position_embeddings': 262144}}))
            for name, port in serve.PORTS.items():
                command, limit = serve.run_command(name, weights, port, 'device=0')
                self.assertEqual(limit, 262144)
                self.assertIn(f'127.0.0.1:{port}:8080', command)
                self.assertIn(f'type=bind,src={weights},dst=/weights,readonly', command)
                self.assertIn('device=0', command)
                self.assertEqual(command[-1], serve.image_name(name))
                if name == 'openjev':
                    self.assertIn('OPENJEV_MAX_MODEL_LEN=262144', command)
                if name == 'openjev_sglang':
                    self.assertIn('OPENJEV_MAX_INPUT_TOKENS=262144', command)
                    self.assertIn('OPENJEV_MAX_TOTAL_INPUT_TOKENS=524288', command)

    def test_build_uses_original_openjev_dockerfiles(self):
        commands = serve.build_commands('djev')
        self.assertEqual(len(commands), 3)
        self.assertIn(str(serve.CACHE / 'openjev/docker/Dockerfile.base'), commands[0])
        self.assertIn(str(serve.CACHE / 'openjev/docker/Dockerfile'), commands[1])
        self.assertIn('BASE_IMAGE=' + serve.image_name('openjev'), commands[2])

    def test_source_verification_rejects_changed_code(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(serve, 'CACHE', Path(tmp)):
            source = Path(tmp) / 'djev'
            source.mkdir()
            (source / '.upstream-commit').write_text(serve.SOURCES['djev']['commit'])
            for filename in serve.SOURCES['djev']['source_sha256']:
                target = source / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'modified upstream source'):
                serve.verify_source('djev')
