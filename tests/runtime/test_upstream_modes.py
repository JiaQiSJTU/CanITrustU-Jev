import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from src.model.diffusion import canvas_width, build_canvas, diffusion_answers
from src.model.open_alternative_jev import LocalLibrary
from src.model.registry import effective_config
from test_reader import example


class UpstreamModeTests(unittest.TestCase):
    def test_canvas_width_rounds_template_and_turn_close(self):
        for template_length, expected in ((1, 16), (15, 16), (16, 32), (31, 32), (32, 48), (63, 64)):
            with self.subTest(template_length=template_length):
                template = [7] * template_length
                width = canvas_width(template, 64)
                self.assertEqual(width, expected)
                canvas = build_canvas(template, 0, width, 262144, 0)
                self.assertEqual(len(canvas), expected)
                self.assertEqual(canvas[template_length], 106)
        with self.assertRaises(ValueError):
            canvas_width([7] * 64, 64)

    def test_openjev_confidence_does_not_change_choice_or_probabilities(self):
        for draws, expected in (([[.5, .5]], 0), ([[1., 0.]], 1),
                                ([[.8, .2]], 1 + (.8 * math.log(.8) + .2 * math.log(.2)) / math.log(2))):
            old, _ = diffusion_answers('choice', ['a', 'b'], draws)
            new, _ = diffusion_answers('choice', ['a', 'b'], draws, 'normalized_entropy')
            self.assertAlmostEqual(new['confidence'], expected)
            self.assertEqual(new['choice'], old['choice'])
            self.assertEqual(new['probabilities'], old['probabilities'])
        answer, _ = diffusion_answers('noul', None, [[.8, .2]], 'normalized_entropy')
        self.assertEqual(answer, {'type': 'noul', 'noul': .8})

    def test_packed_prompt_is_both_checked_and_scored(self):
        record = example()
        count = len(record['options'])
        result = NS(index=0, probabilities=[1.] + [0.] * (count - 1), confidence=1.)
        packed = Mock(return_value=NS(ids=[1, 2, 3]))
        engine = NS(backend=NS(model=NS(config=NS(max_position_embeddings=100))),
                    prompts=NS(packed=packed), decide=Mock(return_value=[result]))
        load = Mock(return_value=engine)
        module = NS(Choice=lambda *a, **kw: NS(), Decider=NS(from_pretrained=load))
        with patch.dict('sys.modules', {'so1': module}):
            adapter = LocalLibrary({'weights': 'fixture'})
        self.assertEqual(load.call_args.kwargs['mode'], 'packed')
        pred = adapter.predict(record)
        packed.assert_called_once()
        self.assertEqual(engine.decide.call_args.kwargs['mode'], 'packed')
        self.assertEqual(pred.raw['backend_metadata']['mode'], 'packed')
        adapter.context['max_input_tokens'] = 2
        engine.decide.reset_mock()
        with self.assertRaises(ValueError):
            adapter.predict(record)
        engine.decide.assert_not_called()

    def test_method_policies_are_in_resume_identity(self):
        cfg = {'models': {'djev': {'backend': 'diffusion'}, 'openjev': {'backend': 'diffusion'}}}
        self.assertEqual(effective_config('djev', cfg)['confidence_method'], 'max_probability')
        openjev = effective_config('openjev', cfg)
        self.assertEqual(openjev['confidence_method'], 'normalized_entropy')
        self.assertEqual(openjev['canvas_step'], 16)
