"""Kev evaluation adapter."""
from .model_utils import finish, prefer_env, resolve_hub_snapshot, local_context
from src.dataset.schema import request
import time
from collections import Counter



def pin_local_base(meta, base):
    """Point a kev checkpoint at a local base directory instead of its Hub revision."""
    meta.base = base
    meta.base_revision = None
    return meta

def encode_kev_with_budget(model, tok, rec, max_state, max_branch):
    """Reserve complete question branches within Kev's state + branch row limit."""
    probe = model.encode(tok, {**rec, 'state': ''}, max_state=1, max_branch=max_branch)
    counts = Counter(probe['seg'])
    branch_tokens = max((n for segment, n in counts.items() if segment != 0), default=0)
    state_budget = min(max_state, max_branch - branch_tokens)
    encoded = model.encode(tok, rec, max_state=state_budget, max_branch=max_branch)
    return encoded, {'row_limit': max_branch, 'branch_tokens': branch_tokens,
                     'state_budget': state_budget,
                     'state_truncated': bool(encoded['state_truncated'])}

class Kev:
    def __init__(self, config):
        from kev.api import SystemOneRequest, output_tokens, to_answers, to_record
        from kev.checkpoint import Checkpoint, LoadOptions, load
        from kev.device import default_device
        self.SystemOneRequest = SystemOneRequest
        self.to_record = to_record
        self.to_answers = to_answers
        self.output_tokens = output_tokens
        self.model_id = config.get('request_model', 'kev-0.6b')
        self.weights = prefer_env('KEV_WEIGHTS', config['weights'])
        self.base = prefer_env('KEV_BASE', config.get('base') or '')
        device = config.get('device') or default_device()
        opts = LoadOptions()
        if self.base:
            checkpoint = Checkpoint(self.weights)
            self.base = pin_local_base(checkpoint.meta, resolve_hub_snapshot(self.base)).base
            self.tok, self.model = checkpoint.load(device, opts)
        else:
            self.tok, self.model = load(self.weights, device, opts)
        self._configure_context(config.get('max_len'))

    def _configure_context(self, max_len=None):
        self.context = local_context(self.model.lm.config, requested=max_len)
        self.max_state = self.max_branch = self.context['max_input_tokens']
        # Leave upstream SERVE_MAX_PACKED unchanged: longer requests automatically
        # use causal rows instead of allocating a packed L x L branch mask.

    def predict(self, record):
        body = request(record, self.model_id)
        req = self.SystemOneRequest(state=body['state'], model=self.model_id, questions=body['questions'])
        rec, meta = self.to_record(req)
        start = time.perf_counter()
        encoded, context = encode_kev_with_budget(
            self.model, self.tok, rec, self.max_state, self.max_branch)
        raw_probs = self.model.probs(encoded)
        lists = [p.tolist() if hasattr(p, 'tolist') else list(p) for p in raw_probs]
        answers = self.to_answers(lists, meta)
        payload = {'model': self.weights, 'answers': answers,
                   'usage': {'input_tokens': len(encoded['ids']),
                             'output_tokens': self.output_tokens(self.tok, answers)},
                   'backend_metadata': {'weights': self.weights, 'base': getattr(self, 'base', ''),
                                        'context': context}}
        pred = finish(payload, record, start)
        if context['state_truncated']:
            pred.truncation = context
        return pred
