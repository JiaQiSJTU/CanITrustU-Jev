"""Jeff evaluation adapter."""
from .model_utils import NotApplicable, finish, prefer_env, local_context
from src.dataset.schema import request
import time



class Jeff:
    def __init__(self, config):
        from jev_clf.client import Choice, Noul, SystemOneClient
        from jev_clf import readout, schema
        self.readout = readout
        self.label_space = schema.label_space
        self.Choice = Choice
        self.Noul = Noul
        self.model_id = config.get('request_model', 'jeff')
        self.client = SystemOneClient(
            base_model=prefer_env('JEFF_BASE', config.get('base_model', 'Qwen/Qwen3-4B-Instruct-2507')),
            adapter=prefer_env('JEFF_WEIGHTS', config.get('weights', 'GestaltLabs/Jeff-1')),
            device=config.get('device'))
        self.context = local_context(self.client.model.config, requested=config.get('max_tokens'))
        self.client.max_length = self.context['max_input_tokens']

    def _question_context(self, question):
        variants = self.readout.label_token_variants(self.client.tokenizer, self.label_space(question))
        reserve = max(map(len, variants.values())) if self.readout.choose_mode(variants) == 'sequence' else 0
        context = dict(self.context)
        context['reserved_tokens'] = reserve
        context['max_input_tokens'] = min(context['max_input_tokens'], context['backbone_max_tokens'] - reserve)
        if context['max_input_tokens'] <= 0:
            raise ValueError('Jeff answer labels exceed the backbone context limit')
        return context

    def predict(self, record):
        body = request(record, self.model_id)
        question = body['questions']['decision']
        if question['type'] == 'noul':
            built = self.Noul(instructions=question['instructions'])
        elif question['type'] == 'choice':
            built = self.Choice(instructions=question['instructions'], criteria=question['criteria'])
        else:
            raise NotApplicable('Jeff adapter supports choice and noul')
        context = self._question_context(built)
        self.client.max_length = context['max_input_tokens']
        start = time.perf_counter()
        result = self.client.system_one(body['state'], {'decision': built})
        if question['type'] == 'noul':
            answer = {'type': 'noul', 'noul': float(result.nouls['decision'].noul)}
        else:
            choice = result.choices['decision']
            answer = {'type': 'choice', 'choice': choice.choice,
                      'probabilities': {k: float(v) for k, v in choice.probabilities.items()},
                      'confidence': float(choice.confidence)}
        payload = {'model': result.model, 'answers': {'decision': answer},
                   'backend_metadata': {'n_forward_passes': result.n_forward_passes,
                                        'context': context,
                                        'readout_modes': dict(result.readout_modes)}}
        return finish(payload, record, start)
