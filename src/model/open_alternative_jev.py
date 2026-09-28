"""open_alternative_jev native evaluation adapter."""
import json
import time
from .model_utils import CallFailure, parse_response, strict_json, NotApplicable, prefer_env, local_context, check_input_tokens, as_text


class LocalLibrary:
    def __init__(self, config):
        from so1 import Decider, Choice
        self.Choice = Choice
        self.model = prefer_env('SO1_WEIGHTS', config['weights'])
        self.mode = config.get('mode', 'packed')
        if self.mode not in ('packed', 'separate'):
            raise ValueError('so1 mode must be packed or separate')
        self.engine = Decider.from_pretrained(self.model, backend='hf', mode=self.mode, **config.get('load_kwargs', {}))
        self.context = local_context(self.engine.backend.model.config, requested=config.get('max_tokens'))

    def predict(self, record):
        if record.get('primitive') == 'noul':
            raise NotApplicable('so1 adapter requires choice records')
        # Native Choice accepts option text; IDs belong only to the result map.
        texts = [o['id'] if o['description'] is None else as_text(o['description']) for o in record['options']]
        state = record['state'] if isinstance(record['state'], str) else json.dumps(record['state'], ensure_ascii=False)
        start = time.perf_counter()
        questions = [self.Choice(record['instructions'], texts, name='decision')]
        prompts = ([self.engine.prompts.packed(state, questions)] if self.mode == 'packed'
                   else self.engine.prompts.separate(state, questions))
        for prompt in prompts:
            check_input_tokens(len(prompt.ids), self.context)
        d = self.engine.decide(state=state, questions=questions, mode=self.mode)[0]
        snapshot = {'mode': self.mode, 'index': getattr(d, 'index', None),
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
