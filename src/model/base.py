from dataclasses import dataclass, field
import json
import math
from typing import Dict, Optional

@dataclass
class Prediction:
    choice: str
    probabilities: Dict[str, float]
    confidence: Optional[float] = None
    elapsed_seconds: float = 0.0
    served_model: str = ''
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    response_text: str = ''
    response_status: Optional[int] = None
    response_headers: dict = field(default_factory=dict)

def parse_response(payload, options, elapsed=0.0):
    answer = payload['answers']['decision']
    ids = [o['id'] for o in options]
    if answer['type'] == 'noul':
        if set(ids) != {'true', 'false'}:
            raise ValueError('Noul response for non-binary question')
        yes = float(answer['noul'])
        probs = {'true': yes, 'false': 1 - yes}
        choice = 'true' if yes >= .5 else 'false'
    elif answer['type'] == 'choice':
        probs = {k: float(v) for k, v in answer['probabilities'].items()}
        choice = answer['choice']
    else:
        raise ValueError('Unexpected answer type')
    if set(probs) != set(ids) or choice not in probs:
        raise ValueError('Response must contain exactly the requested options')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()):
        raise ValueError('Invalid probabilities')
    if abs(sum(probs.values()) - 1) > 1e-4:
        raise ValueError('Probabilities do not sum to one; no silent renormalization')
    if probs[choice] + 1e-6 < max(probs.values()):
        raise ValueError('Choice is not a maximum-probability option')
    confidence = answer.get('confidence')
    if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError('Invalid provider confidence')
    return Prediction(choice, probs, confidence, elapsed, payload.get('model', ''), payload.get('usage', {}), payload)


class NotApplicable(ValueError):
    """The selected backend cannot represent this decision."""


class CallFailure(Exception):
    """Provider call finished, but the body cannot be scored. The body is retained."""

    def __init__(self, message, response_text='', response=None, response_status=None, response_headers=None):
        super().__init__(message)
        self.response_text = response_text or ''
        self.response = response
        self.response_status = response_status
        self.response_headers = response_headers or {}


def parse_json_or_none(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def strict_json(value):
    """JSON-safe copy. Non-finite floats stay recoverable instead of dropping the row."""
    if value is None or isinstance(value, str) or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {'__nonfinite__': 'NaN'}
        if math.isinf(value):
            return {'__nonfinite__': 'Infinity' if value > 0 else '-Infinity'}
        return value
    if isinstance(value, dict):
        return {str(k): strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    return str(value)
