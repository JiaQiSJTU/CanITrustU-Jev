import copy
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.dataset.schema import request
from src.model.context import estimate_tokens, fit_record
from src.model.jev import Jev
from src.model.registry import load_model
from test_adapters import Body
from test_reader import example


def _messages(parts, target):
    record = example()
    record['state'] = {'messages': [{'role': 'user', 'content': text} for text in parts],
                       'target_message_index': target, 'tools': [{'name': 'lookup'}]}
    return record


def _tokens(record):
    return estimate_tokens(json.dumps(request(record, 'jev-1.13.0'), ensure_ascii=False))


class ContextTests(unittest.TestCase):
    def test_short_state_is_not_copied(self):
        record = example()
        fitted, info = fit_record(record, 32000, lambda rec: request(rec, 'jev-1.13.0'))
        self.assertIs(fitted, record)
        self.assertIsNone(info)

    def test_drops_early_messages_and_keeps_the_anchor(self):
        record = _messages(['EARLY ' * 8000, 'MID', 'LATE'], 2)
        kept = _messages(['MID', 'LATE'], 1)
        fitted, info = fit_record(record, _tokens(kept), lambda rec: request(rec, 'jev-1.13.0'))
        self.assertEqual([m['content'] for m in fitted['state']['messages']], ['MID', 'LATE'])
        self.assertEqual(fitted['state']['target_message_index'], 1)
        self.assertEqual(fitted['state']['tools'], [{'name': 'lookup'}])
        self.assertEqual(record['state']['target_message_index'], 2)
        self.assertIn('EARLY', record['state']['messages'][0]['content'])
        self.assertGreater(info['dropped_history_items'], 0)
        self.assertLessEqual(info['estimated_tokens_after'], info['max_input_tokens'])

    def test_left_truncates_the_anchor_message_itself(self):
        record = _messages(['HEAD' + ('x' * 20000) + 'TAIL'], 0)
        kept = _messages(['TAIL'], 0)
        fitted, info = fit_record(record, _tokens(kept), lambda rec: request(rec, 'jev-1.13.0'))
        content = fitted['state']['messages'][0]['content']
        self.assertTrue(content.endswith('TAIL'))
        self.assertNotIn('HEAD', content)
        self.assertEqual(fitted['state']['target_message_index'], 0)
        self.assertGreater(info['trimmed_chars'], 0)

    def test_json_string_state_and_nested_task_input(self):
        record = _messages(['EARLY ' * 8000, 'LATE'], 1)
        record['state'] = json.dumps(record['state'], ensure_ascii=False, indent=2)
        fitted, info = fit_record(record, _tokens(_messages(['LATE'], 0)), lambda rec: request(rec, 'jev-1.13.0'))
        self.assertIsInstance(fitted['state'], str)
        state = json.loads(fitted['state'])
        self.assertEqual(state['messages'][0]['content'], 'LATE')
        self.assertEqual(state['target_message_index'], 0)
        self.assertGreater(info['dropped_history_items'], 0)

        nested = example()
        nested['state'] = {'task_input': {'messages': [{'role': 'user', 'content': 'EARLY ' * 8000},
                                                       {'role': 'assistant', 'content': 'LATE'}],
                                          'target_message_index': 1},
                           'auxiliary_context': {'content': 'AUX stays'}}
        budget = example()
        budget['state'] = {'task_input': {'messages': [{'role': 'assistant', 'content': 'LATE'}],
                                          'target_message_index': 0},
                           'auxiliary_context': {'content': 'AUX stays'}}
        fitted, info = fit_record(nested, _tokens(budget), lambda rec: request(rec, 'jev-1.13.0'))
        self.assertEqual(fitted['state']['task_input']['messages'][0]['content'], 'LATE')
        self.assertEqual(fitted['state']['auxiliary_context']['content'], 'AUX stays')
        self.assertEqual(fitted['state']['task_input']['target_message_index'], 0)

    def test_plain_string_keeps_its_suffix(self):
        record = example()
        record['state'] = ('EARLY ' * 8000) + 'SUFFIX'
        kept = copy.deepcopy(record)
        kept['state'] = 'SUFFIX'
        fitted, info = fit_record(record, _tokens(kept), lambda rec: request(rec, 'jev-1.13.0'))
        self.assertTrue(fitted['state'].endswith('SUFFIX'))
        self.assertNotIn('EARLY', fitted['state'])
        self.assertGreater(info['trimmed_chars'], 0)

    def test_predict_sends_the_truncated_state(self):
        record = _messages(['EARLY ' * 4000, 'LATE'], 1)
        body = {'model': 'fixture', 'answers': {'decision': {'type': 'choice', 'choice': 'z',
                'probabilities': {'z': 1., 'a': 0.}}}}
        with patch('src.model.jev.urlopen', return_value=Body(body)) as call:
            pred = Jev(key_env=None, max_input_tokens=_tokens(_messages(['LATE'], 0))).predict(record)
        sent = json.loads(call.call_args.args[0].data)
        self.assertEqual(sent['state']['messages'][0]['content'], 'LATE')
        self.assertEqual(sent['state']['target_message_index'], 0)
        self.assertNotIn('gold', sent)
        self.assertEqual(pred.truncation['dropped_history_items'], 1)
        self.assertLessEqual(estimate_tokens(call.call_args.args[0].data.decode()), pred.truncation['max_input_tokens'])

    def test_jev_config_sets_the_budget(self):
        config = json.loads(Path('configs/models.json').read_text())
        with patch.dict(os.environ, {'JEV_ENDPOINT': 'http://localhost', 'JEV_API_KEY': 'secret'}):
            model, entry = load_model('jev', config)
        self.assertEqual(entry['max_input_tokens'], 32000)
        self.assertEqual(model.max_input_tokens, 32000)


if __name__ == '__main__':
    unittest.main()
