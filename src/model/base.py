from dataclasses import dataclass, field
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
