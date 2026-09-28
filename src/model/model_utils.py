"""Shared evaluation responses, validation, context budgeting and adapter utilities."""
from dataclasses import dataclass, field
from typing import Dict, Optional
import copy
import json
import math
import os
import time
from src.dataset.schema import request


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

DEFAULT_PROBABILITY_DECIMALS = 4


class UpstreamService:
    """Thin client for an unchanged upstream API running on the local host.

    Prompt construction, labels, seeds, inference and confidence all belong to
    the pinned upstream process. This layer only maps the evaluation schema.
    """
    def __init__(self, config):
        from .jev import Jev
        self.client = Jev(endpoint=config['default_endpoint'], model=config['request_model'],
                          key_env=config.get('key_env'), timeout=config.get('timeout', 600))

    def predict(self, record):
        return self.client.predict(record)


def normalized_entropy_confidence(probabilities):
    """OpenJev confidence: zero for uniform, one for a point distribution."""
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0)
    return min(1.0, max(0.0, 1 - entropy / math.log(len(probabilities))))


def local_context(model_config, *, reserve=0, requested=None):
    """Resolve the loaded text backbone limit, excluding method-owned tokens.

    Tokenizer model_max_length can be a sentinel or a training default, so it
    is deliberately not used as the architectural limit. No RoPE extension
    beyond the checkpoint's declared positions is invented here.
    """
    def get(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    text = get(model_config, 'text_config')
    cfg = text if text is not None else model_config
    limits = [get(cfg, key) for key in ('max_position_embeddings', 'n_positions', 'max_seq_len', 'seq_length')]
    limits = [n for n in limits if type(n) is int and 0 < n < 10**9]
    if not limits:
        raise ValueError('Cannot resolve local model context limit from its text configuration')
    limit = min(limits)
    if type(reserve) is not int or not 0 <= reserve < limit:
        raise ValueError('Reserved tokens must be smaller than the backbone context limit')
    available = limit - reserve
    if requested is not None:
        if type(requested) is not int or not 0 < requested <= available:
            raise ValueError(f'Requested context {requested!r} exceeds available limit {available} or is invalid')
        available = requested
    return {'policy': 'model_max', 'backbone_max_tokens': limit,
            'reserved_tokens': reserve, 'max_input_tokens': available}


def check_input_tokens(count, context):
    if count > context['max_input_tokens']:
        raise ValueError(f"Input has {count} tokens; model input limit is {context['max_input_tokens']}")

def probability_sum_tolerance(option_count, decimals=DEFAULT_PROBABILITY_DECIMALS):
    """Shared acceptance budget for independently rounded choice probabilities.

    Four decimals is the default validation policy, not inferred provider
    precision. Adapters can declare a finer precision when it is known.
    """
    if type(option_count) is not int or not 2 <= option_count <= 255:
        raise ValueError('Choice must contain 2..255 options')
    if type(decimals) is not int or not 4 <= decimals <= 15:
        raise ValueError('Probability precision must be an integer from 4 to 15')
    return option_count * 0.5 * 10 ** (-decimals) + 1e-12

def parse_response(payload, options, elapsed=0.0, *, probability_decimals=DEFAULT_PROBABILITY_DECIMALS):
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
    tolerance = 1e-12
    if answer['type'] == 'choice':
        # Independently rounded probabilities accumulate up to half a unit
        # in the final decimal place per option. Preserve the reported values.
        tolerance = probability_sum_tolerance(len(probs), probability_decimals)
    total = math.fsum(probs.values())
    if abs(total - 1) > tolerance:
        raise ValueError(f'Probabilities do not sum to one: sum={total:.12g}, '
                         f'options={len(probs)}, tolerance={tolerance:.12g}; '
                         'no silent renormalization')
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

MAX_INPUT_TOKENS = 32000

_HISTORY_LISTS = ('messages', 'trajectory', 'previous_actions', 'context')

_HISTORY_STRINGS = ('action_history', 'utterances')

def estimate_tokens(text):
    letters = digits = punct = other = 0
    for ch in text:
        if ch.isspace():
            continue
        code = ord(ch)
        if 'A' <= ch <= 'Z' or 'a' <= ch <= 'z':
            letters += 1
        elif ch.isdigit():
            digits += 1
        elif code < 128:
            punct += 1
        else:
            other += 1
    return math.ceil(0.30 * letters + digits + 0.60 * punct + other + 800)

def fit_record(record, max_tokens, payload_of=None):
    """Copy `record` only when its request exceeds `max_tokens`.

    Early history is removed from the left. The anchored message
    (`target_message_index`) and everything after it stay until the prefix
    is gone; leftover text is then cut from the front of the earliest
    remaining history.
    """
    payload_of = payload_of or (lambda rec: request(rec, 'jev-latest'))

    def tokens(rec):
        return estimate_tokens(json.dumps(payload_of(rec), ensure_ascii=False))

    before = tokens(record)
    if before <= max_tokens:
        return record, None
    fitted = copy.deepcopy(record)
    editable, as_json = _unwrap(fitted['state'])
    changed = {'dropped': 0, 'trimmed': 0}

    def under_limit():
        if as_json and not isinstance(editable, str):
            fitted['state'] = json.dumps(editable, ensure_ascii=False, indent=2)
        else:
            fitted['state'] = editable
        return tokens(fitted) <= max_tokens

    if isinstance(editable, str):
        box = [editable]

        def setter(text):
            box[0] = text

        def under():
            fitted['state'] = box[0]
            return tokens(fitted) <= max_tokens

        changed['trimmed'] += _cut(lambda: box[0], setter, under)
        fitted['state'] = box[0]
    elif isinstance(editable, dict):
        _truncate_mapping(editable, under_limit, changed)
        under_limit()
    elif isinstance(editable, list):
        holder = {'messages': editable}

        def under():
            fitted['state'] = holder['messages']
            return tokens(fitted) <= max_tokens

        _shrink_list(holder, 'messages', under, changed)
        fitted['state'] = holder['messages']
    after = tokens(fitted)
    info = {'max_input_tokens': max_tokens, 'estimated_tokens_before': before,
            'estimated_tokens_after': after, 'dropped_history_items': changed['dropped'],
            'trimmed_chars': changed['trimmed']}
    return fitted, info

def _unwrap(state):
    if isinstance(state, str):
        stripped = state.lstrip()
        if stripped.startswith('{') or stripped.startswith('['):
            try:
                parsed = json.loads(state)
            except json.JSONDecodeError:
                return state, False
            if isinstance(parsed, (dict, list)):
                return parsed, True
    return state, False

def _truncate_mapping(state, fits, changed):
    if isinstance(state.get('task_input'), dict):
        _truncate_mapping(state['task_input'], fits, changed)
        if not fits():
            _trim_value(state, fits, changed, skip='task_input')
        return
    for key in _HISTORY_LISTS:
        if isinstance(state.get(key), list) and state[key]:
            _shrink_list(state, key, fits, changed)
            if fits():
                return
    for key in _HISTORY_STRINGS:
        if isinstance(state.get(key), str):
            changed['trimmed'] += _cut_field(state, key, fits)
            if fits():
                return
    if not fits():
        _trim_value(state, fits, changed)

def _shrink_list(container, key, fits, changed):
    if fits():
        return
    items = container.get(key)
    if not isinstance(items, list) or not items:
        return
    index_key = None
    if key == 'messages' and isinstance(container.get('target_message_index'), int):
        index_key = 'target_message_index'
        target = container[index_key]
        if not 0 <= target < len(items):
            target = len(items) - 1
    else:
        target = len(items) - 1
    original = list(items)

    def apply(count):
        container[key] = original[count:]
        if index_key:
            container[index_key] = target - count

    max_drop = target
    if max_drop:
        apply(max_drop)
        if fits():
            lo, hi = 0, max_drop
            while lo < hi:
                mid = (lo + hi) // 2
                apply(mid)
                if fits():
                    hi = mid
                else:
                    lo = mid + 1
            apply(lo)
            changed['dropped'] += lo
        else:
            changed['dropped'] += max_drop
    kept = container[key]
    for index, item in enumerate(kept):
        if fits():
            return
        if isinstance(item, str):
            changed['trimmed'] += _cut(lambda index=index: container[key][index],
                                       lambda value, index=index: container[key].__setitem__(index, value), fits)
        else:
            _shrink_item(item, fits, changed)

def _shrink_item(item, fits, changed):
    if fits() or not isinstance(item, dict):
        return
    if isinstance(item.get('messages'), list):
        _shrink_list(item, 'messages', fits, changed)
    if fits():
        return
    keys = [k for k in item if isinstance(item[k], str)]
    if 'content' in keys:
        keys.remove('content')
        keys.insert(0, 'content')
    for key in keys:
        changed['trimmed'] += _cut_field(item, key, fits)
        if fits():
            return
    for key in item:
        if not isinstance(item[key], str):
            _trim_value(item[key], fits, changed)
            if fits():
                return

def _trim_value(value, fits, changed, skip=None):
    if fits():
        return
    if isinstance(value, dict):
        for key in list(value.keys()):
            if key == skip:
                continue
            child = value[key]
            if isinstance(child, str):
                changed['trimmed'] += _cut_field(value, key, fits)
            else:
                _trim_value(child, fits, changed)
            if fits():
                return
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, str):
                changed['trimmed'] += _cut(lambda index=index: value[index],
                                           lambda text, index=index: value.__setitem__(index, text), fits)
            else:
                _trim_value(child, fits, changed)
            if fits():
                return

def _cut_field(container, key, fits):
    return _cut(lambda: container[key], lambda text: container.__setitem__(key, text), fits)

def _cut(getter, setter, fits):
    """Keep the longest suffix of a string that satisfies `fits`."""
    text = getter()
    if not isinstance(text, str) or not text:
        return 0
    setter('')
    if not fits():
        setter(text)
        return 0
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        setter(text[mid:])
        if fits():
            hi = mid
        else:
            lo = mid + 1
    setter(text[lo:])
    return lo

def prefer_env(name, fallback):
    value = os.environ.get(name, '')
    return value if value else fallback

def resolve_hub_snapshot(path):
    """A model directory, or a Hugging Face `snapshots` folder resolved through `refs/main`."""
    path = os.path.abspath(path)
    if os.path.isfile(os.path.join(path, 'config.json')):
        return path
    if os.path.basename(path) == 'snapshots':
        ref = os.path.join(os.path.dirname(path), 'refs', 'main')
        if os.path.isfile(ref):
            with open(ref, encoding='utf-8') as handle:
                pinned = os.path.join(path, handle.read().strip())
            if os.path.isdir(pinned):
                return pinned
    raise ValueError('Not a local model snapshot or snapshots directory: ' + path)

def finish(payload, record, started, *, probability_decimals=DEFAULT_PROBABILITY_DECIMALS):
    try:
        pred = parse_response(payload, record['options'], time.perf_counter() - started,
                              probability_decimals=probability_decimals)
    except Exception as e:
        text = json.dumps(strict_json(payload), ensure_ascii=False)
        raise CallFailure(str(e), text, payload) from None
    pred.response_text = json.dumps(strict_json(payload), ensure_ascii=False)
    return pred

def _redact(value, secret):
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, '[redacted]')
    if isinstance(value, list):
        return [_redact(v, secret) for v in value]
    if isinstance(value, dict):
        return {_redact(k, secret) if isinstance(k, str) else k: _redact(v, secret) for k, v in value.items()}
    return value

_HIDDEN_HEADERS = {'authorization', 'cookie', 'set-cookie', 'proxy-authorization', 'x-api-key'}


def _header_map(headers, secret):
    if headers is None:
        return {}
    kept = {}
    for key, value in headers.items():
        if str(key).lower() in _HIDDEN_HEADERS:
            continue
        kept[str(key)] = _redact(value, secret)
    return kept

def _decode_return(body, secret):
    body = body or b''
    text = body.decode('utf-8', errors='replace')
    parsed = parse_json_or_none(text)
    if parsed is not None:
        parsed = _redact(parsed, secret)
    return _redact(text, secret), parsed

def encode(tok, text):
    return [int(t) for t in tok.encode(text, add_special_tokens=False)]

def as_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
