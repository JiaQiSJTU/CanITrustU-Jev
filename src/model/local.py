"""Optional native library integration; weights are never installed by this project."""
import json
import time
from .base import CallFailure, parse_response, strict_json, NotApplicable

class LocalLibrary:
    def __init__(self, config):
        from so1 import Decider, Choice
        self.Choice = Choice
        self.model = config['weights']
        self.engine = Decider.from_pretrained(self.model, backend='hf', mode='separate', **config.get('load_kwargs', {}))

    def predict(self, record):
        if record.get('primitive') == 'noul':
            raise NotApplicable('so1 adapter requires choice records')
        texts = [json.dumps({'id': o['id'], 'description': o['description']}, ensure_ascii=False) for o in record['options']]
        state = record['state'] if isinstance(record['state'], str) else json.dumps(record['state'], ensure_ascii=False)
        start = time.perf_counter()
        d = self.engine.decide(state=state, questions=[self.Choice(record['instructions'], texts, name='decision')], mode='separate')[0]
        snapshot = {'index': getattr(d, 'index', None),
                    'probabilities': [float(p) for p in getattr(d, 'probabilities', [])],
                    'confidence': None if getattr(d, 'confidence', None) is None else float(d.confidence)}
        if len(d.probabilities) != len(texts):
            raise CallFailure('so1 returned wrong probability count', response=snapshot)
        answer = {'type': 'choice', 'choice': record['options'][d.index]['id'],
                  'probabilities': {o['id']: float(p) for o, p in zip(record['options'], d.probabilities)}, 'confidence': float(d.confidence)}
        payload = {'model': self.model, 'answers': {'decision': answer}, 'backend_metadata': snapshot}
        try:
            pred = parse_response(payload, record['options'], time.perf_counter() - start)
        except Exception as e:
            raise CallFailure(str(e), response=payload) from None
        pred.response_text = json.dumps(strict_json(payload), ensure_ascii=False)
        return pred

class SemIf:
    def __init__(self, config):
        from semif_phase1.core import load_causal_model
        from semif_phase1.direct import score
        self.score = score
        self.model_id = config['request_model']
        self.model, self.tokenizer, self.metadata = load_causal_model(
            self.model_id, config['weights_revision'], config.get('device', 'auto'), config.get('dtype', 'bfloat16'))
        self.max_tokens = config.get('max_tokens', 4096)

    def predict(self, record):
        if record.get('primitive') == 'noul':
            raise NotApplicable('SemIf native direct adapter requires choice records')
        row = {'id': 'decision', 'state': record['state'], 'question': record['instructions'],
               'options': [{'id': o['id'], 'description': o['description'] if isinstance(o['description'], str)
                            else json.dumps(o['description'], ensure_ascii=False)} for o in record['options']]}
        start = time.perf_counter()
        out = self.score(self.model, self.tokenizer, row, self.metadata, self.max_tokens)
        if out['option_ids'] != [o['id'] for o in row['options']] or len(out['probabilities']) != len(row['options']):
            raise CallFailure('SemIf option alignment mismatch', response=out)
        probs = dict(zip(out['option_ids'], map(float, out['probabilities'])))
        answer = {'type': 'choice', 'choice': max(probs, key=probs.get), 'probabilities': probs}
        payload = {'model': self.model_id, 'answers': {'decision': answer}, 'usage': {'input_tokens': out['input_tokens']},
                   'backend_metadata': out}
        try:
            pred = parse_response(payload, record['options'], time.perf_counter() - start)
        except Exception as e:
            raise CallFailure(str(e), response=payload) from None
        pred.response_text = json.dumps(strict_json(payload), ensure_ascii=False)
        return pred
