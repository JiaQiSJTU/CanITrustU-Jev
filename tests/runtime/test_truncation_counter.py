import copy
import json
import random
import unittest

from src.dataset.schema import request
from src.model.model_utils import (_JsonTokenCounter, _estimated_tokens,
                                   estimate_tokens, fit_record)
from test_reader import example


class TruncationCounterTests(unittest.TestCase):
    def test_counts_match_serialized_json_including_embedded_json(self):
        rng = random.Random(719)
        atoms = [None, True, False, 0, -123, 1.234e-7, '', '汉字 ² ٣ é😀',
                 '\\"\n\t\r\b\f\x00\x1f', 'abc123!\u2028\u00a0']

        def tree(depth):
            if not depth or rng.randrange(3) == 0:
                return rng.choice(atoms)
            if rng.randrange(2):
                return [tree(depth - 1) for _ in range(rng.randrange(5))]
            return {str(i) + '\\"\n': tree(depth - 1) for i in range(rng.randrange(5))}

        counter = _JsonTokenCounter()
        values = atoms + [{1: 'x', None: [True]}, [], {}] + [tree(4) for _ in range(100)]
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(_estimated_tokens(counter.count(value)),
                                 estimate_tokens(json.dumps(value, ensure_ascii=False)))
                counts = counter.count(value, embedded=True)
                counts = (counts[0], counts[1], counts[2] + 2, counts[3])
                wire = json.dumps(json.dumps(value, ensure_ascii=False, indent=2), ensure_ascii=False)
                self.assertEqual(_estimated_tokens(counts), estimate_tokens(wire))

    def test_many_fields_shrink_cumulatively_without_rebuilding_payload(self):
        r = example()
        r['state'] = {str(i): 'x' * 2000 for i in range(80)}
        original = copy.deepcopy(r)
        calls = []

        def payload(rec):
            calls.append(1)
            return request(rec)

        fitted, info = fit_record(r, 2000, payload)
        actual = estimate_tokens(json.dumps(request(fitted), ensure_ascii=False))
        self.assertEqual(actual, info['estimated_tokens_after'])
        self.assertLessEqual(actual, 2000)
        self.assertGreater(info['trimmed_chars'], 100000)
        self.assertLessEqual(len(calls), 3)
        self.assertEqual(r, original)
        self.assertEqual(fitted['options'], r['options'])
        self.assertEqual(fitted['instructions'], r['instructions'])

    def test_json_list_string_preserves_representation_and_suffix(self):
        r = example()
        r['state'] = json.dumps(['early ' * 4000, 'LATE'])
        kept = {**r, 'state': json.dumps(['LATE'], indent=2)}
        budget = estimate_tokens(json.dumps(request(kept), ensure_ascii=False))
        fitted, info = fit_record(r, budget)
        self.assertIsInstance(fitted['state'], str)
        self.assertEqual(json.loads(fitted['state']), ['LATE'])
        self.assertLessEqual(info['estimated_tokens_after'], budget)

    def test_custom_state_transformation_falls_back_to_complete_payload(self):
        r = example()
        r['state'] = 'x' * 2000 + 'TAIL'

        def payload(rec):
            return {'state': 'prefix:' + rec['state'], 'duplicate': rec['state']}

        fitted, info = fit_record(r, 900, payload)
        self.assertEqual(info['estimated_tokens_after'],
                         estimate_tokens(json.dumps(payload(fitted), ensure_ascii=False)))
        self.assertLessEqual(info['estimated_tokens_after'], 900)
        self.assertTrue(fitted['state'].endswith('TAIL'))
