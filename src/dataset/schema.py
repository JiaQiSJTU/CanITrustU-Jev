"""Versioned decision records; gold/provenance never enter model requests."""
import copy
import hashlib
import json

SCHEMA_VERSION = 'jev-choice/1.0'

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def validate(record):
    if record.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported schema_version')
    for key in ('id', 'dataset', 'group_id', 'state', 'instructions', 'options', 'gold', 'source'):
        if key not in record:
            raise ValueError('Missing ' + key)
    options = record['options']
    if not 2 <= len(options) <= 255:
        raise ValueError('Choice must contain 2..255 options')
    ids = [o['id'] for o in options]
    if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError('Option IDs must be nonempty and unique')
    if record['gold'] not in ids:
        raise ValueError('Gold absent from options')
    for o in options:
        if not o.get('canonical_id'):
            raise ValueError('Missing canonical ID')
    if len({o['canonical_id'] for o in options}) != len(options):
        raise ValueError('Duplicate canonical ID')
    if record.get('primitive', 'choice') == 'noul' and set(ids) != {'true', 'false'}:
        raise ValueError('Noul requires true/false')
    json.dumps(record, allow_nan=False)
    return record

def model_input(record):
    """Explicit allowlist: never send IDs, splits, gold or annotations."""
    validate(record)
    return copy.deepcopy({k: record[k] for k in ('state', 'instructions', 'options')})

def request(record, model='jev-latest', primitive=None):
    data = model_input(record)
    kind = primitive or record.get('primitive', 'choice')
    if kind not in ('choice', 'noul'):
        raise ValueError('Classification runner supports choice/noul only')
    criteria = {o['id']: o['description'] for o in data['options']}
    if kind == 'noul' and set(criteria) != {'true', 'false'}:
        raise ValueError('Noul requires binary labels')
    return {'model': model, 'state': data['state'], 'questions': {'decision': {
        'type': kind, 'instructions': data['instructions'], 'criteria': criteria}}}
