"""semif native evaluation adapter."""
import json
import os
import time
from .model_utils import CallFailure, parse_response, strict_json, NotApplicable, prefer_env, local_context


class SemIf:
    def __init__(self, config):
        from semif_phase1.core import load_causal_model
        from semif_phase1.direct import score
        self.score = score
        self.model_id = prefer_env('SEMIF_MODEL', config['request_model'])
        revision = os.environ['SEMIF_REVISION'] if 'SEMIF_REVISION' in os.environ else config['weights_revision']
        self.model, self.tokenizer, self.metadata = load_causal_model(
            self.model_id, revision or None, config.get('device', 'auto'), config.get('dtype', 'bfloat16'))
        self.context = local_context(self.model.config, requested=config.get('max_tokens'))
        self.max_tokens = self.context['max_input_tokens']

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
