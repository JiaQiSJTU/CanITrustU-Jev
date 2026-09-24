"""Left-truncate early history so a Jev request stays inside its input budget."""
import copy
import json
import math

from src.dataset.schema import request

# jev-1.13: state plus the longest question must stay within 32k tokens.
# This estimate is an upper bound on usage.input_tokens for that request shape:
# letters pack at about 3.3 characters per token, digits and punctuation do not.
# The pad keeps the estimate above the provider count on calibrated traffic.
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
