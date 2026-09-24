"""In-process weight loading for open decision models. No HTTP client is used."""
import json
import time
from .base import CallFailure, NotApplicable, parse_response, strict_json
from src.dataset.schema import request


def finish(payload, record, started):
    try:
        pred = parse_response(payload, record['options'], time.perf_counter() - started)
    except Exception as e:
        text = json.dumps(strict_json(payload), ensure_ascii=False)
        raise CallFailure(str(e), text, payload) from None
    pred.response_text = json.dumps(strict_json(payload), ensure_ascii=False)
    return pred


class Laya:
    def __init__(self, config):
        import laya
        self.model_id = config.get('weights', 'convaiinnovations/laya')
        self.agent = laya.load(self.model_id, device=config.get('device'), subfolder=config.get('subfolder'))

    def predict(self, record):
        body = request(record, self.model_id)
        start = time.perf_counter()
        out = self.agent.predict(body['state'], body['questions'])
        if not isinstance(out, dict) or 'answers' not in out:
            raise CallFailure('Laya returned no answers', response=out)
        payload = {'model': out.get('model', self.model_id), **{k: v for k, v in out.items() if k != 'model'}}
        return finish(payload, record, start)


class Jeff:
    def __init__(self, config):
        from jev_clf.client import Choice, Noul, SystemOneClient
        self.Choice = Choice
        self.Noul = Noul
        self.model_id = config.get('request_model', 'jeff')
        self.client = SystemOneClient(
            base_model=config.get('base_model', 'Qwen/Qwen3-4B-Instruct-2507'),
            adapter=config.get('weights', 'GestaltLabs/Jeff-1'),
            device=config.get('device'))

    def predict(self, record):
        body = request(record, self.model_id)
        question = body['questions']['decision']
        if question['type'] == 'noul':
            built = self.Noul(instructions=question['instructions'])
        elif question['type'] == 'choice':
            built = self.Choice(instructions=question['instructions'], criteria=question['criteria'])
        else:
            raise NotApplicable('Jeff adapter supports choice and noul')
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
                                        'readout_modes': dict(result.readout_modes)}}
        return finish(payload, record, start)


class Kev:
    def __init__(self, config):
        from kev.api import SystemOneRequest, output_tokens, to_answers, to_record
        from kev.checkpoint import LoadOptions, load
        from kev.device import default_device
        from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE
        self.SystemOneRequest = SystemOneRequest
        self.to_record = to_record
        self.to_answers = to_answers
        self.output_tokens = output_tokens
        self.max_state = SERVE_MAX_STATE
        self.max_branch = SERVE_MAX_BRANCH
        self.model_id = config.get('request_model', 'kev-0.6b')
        self.weights = config['weights']
        self.tok, self.model = load(self.weights, config.get('device') or default_device(), LoadOptions())

    def predict(self, record):
        body = request(record, self.model_id)
        req = self.SystemOneRequest(state=body['state'], model=self.model_id, questions=body['questions'])
        rec, meta = self.to_record(req)
        start = time.perf_counter()
        encoded = self.model.encode(self.tok, rec, max_state=self.max_state, max_branch=self.max_branch)
        raw_probs = self.model.probs(encoded)
        lists = [p.tolist() if hasattr(p, 'tolist') else list(p) for p in raw_probs]
        answers = self.to_answers(lists, meta)
        payload = {'model': self.weights, 'answers': answers,
                   'usage': {'input_tokens': len(encoded['ids']),
                             'output_tokens': self.output_tokens(self.tok, answers)},
                   'backend_metadata': {'weights': self.weights}}
        return finish(payload, record, start)
