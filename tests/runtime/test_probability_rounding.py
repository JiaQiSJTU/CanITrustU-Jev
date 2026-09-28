import copy
import unittest
from types import SimpleNamespace

from src.model.model_utils import CallFailure, parse_response, probability_sum_tolerance
from src.model.laya import Laya


class ProbabilityRoundingTests(unittest.TestCase):
    def test_laya_rounding_budget_and_validation(self):
        options = [{'id': str(i), 'canonical_id': str(i), 'description': str(i)} for i in range(16)]
        probs = {o['id']: 0.0625 for o in options}
        probs['0'] = 0.0629
        payload = {'answers': {'decision': {
            'type': 'choice', 'choice': '0', 'probabilities': probs}}}
        with self.assertRaises(ValueError):
            parse_response(payload, options, probability_decimals=6)
        self.assertEqual(parse_response(payload, options).probabilities, probs)
        pred = parse_response(payload, options, probability_decimals=4)
        self.assertEqual(pred.probabilities, probs)
        model = Laya.__new__(Laya)
        model.model_id = 'fixture'
        model.agent = SimpleNamespace(predict=lambda *args: payload)
        record = {'schema_version': 'jev-choice/1.0', 'id': 'test',
                  'dataset': 'test', 'group_id': 'test', 'gold': '0', 'source': {},
                  'state': 'test', 'instructions': 'choose', 'options': options}
        self.assertEqual(model.predict(record).choice, '0')
        for mutation in ('sum', 'nan', 'missing', 'choice'):
            bad = copy.deepcopy(payload)
            answer = bad['answers']['decision']
            if mutation == 'sum':
                answer['probabilities']['0'] = 0.064
            elif mutation == 'nan':
                answer['probabilities']['0'] = float('nan')
            elif mutation == 'missing':
                del answer['probabilities']['1']
            else:
                answer['choice'] = '1'
            model.agent.predict = lambda *args: bad
            with self.assertRaises(CallFailure):
                model.predict(record)

    def test_three_option_float_boundary(self):
        probs = {'a': 0.1843, 'b': 0.5277, 'c': 0.2879}
        payload = {'answers': {'decision': {
            'type': 'choice', 'choice': 'b', 'probabilities': probs}}}
        self.assertEqual(parse_response(payload, [{'id': k} for k in probs],
                                        probability_decimals=4).choice, 'b')

    def test_shared_budget_boundaries(self):
        for count in (2, 3, 16, 255):
            options = [{'id': str(i)} for i in range(count)]
            budget = count * 0.00005
            for sign in (-1, 1):
                for within in (True, False):
                    with self.subTest(count=count, sign=sign, within=within):
                        value = (1 + sign * (budget if within else budget + 1e-8)) / count
                        payload = {'answers': {'decision': {'type': 'choice', 'choice': '0',
                                   'probabilities': {o['id']: value for o in options}}}}
                        if within:
                            pred = parse_response(payload, options)
                            self.assertEqual(pred.raw, payload)
                            self.assertEqual(pred.probabilities['0'], value)
                        else:
                            with self.assertRaisesRegex(ValueError, 'options=.*tolerance='):
                                parse_response(payload, options)

    def test_invalid_precision_policy(self):
        for decimals in (None, True, -1, 0, 3, 4.5, 16):
            with self.subTest(decimals=decimals), self.assertRaises(ValueError):
                probability_sum_tolerance(3, decimals)
