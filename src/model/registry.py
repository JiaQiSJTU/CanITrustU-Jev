"""Instantiate configured evaluation adapters."""
import os
from .mock import Mock
from .jev import Jev, SystemOneOpen

def load_model(name, config):
    if name == 'mock':
        return Mock(), {'backend': 'mock', 'purpose': 'pipeline test only'}
    entry = config['models'][name]
    if entry['backend'] == 'python':
        from src.model.local import LocalLibrary
        return LocalLibrary(entry), entry
    if entry['backend'] == 'semif':
        from src.model.local import SemIf
        return SemIf(entry), entry
    if entry['backend'] == 'bridge':
        from src.model.bridge import Bridge
        return Bridge(entry), entry
    if entry['backend'] == 'laya':
        from src.model.weights import Laya
        return Laya(entry), entry
    if entry['backend'] == 'jeff':
        from src.model.weights import Jeff
        return Jeff(entry), entry
    if entry['backend'] == 'kev':
        from src.model.weights import Kev
        return Kev(entry), entry
    endpoint = os.getenv(entry['endpoint_env'], entry.get('default_endpoint', ''))
    if not endpoint:
        raise ValueError('Set endpoint environment variable ' + entry['endpoint_env'])
    adapter = SystemOneOpen if entry['backend'] == 'system_one_http' else Jev
    return adapter(endpoint=endpoint, model=entry['request_model'], key_env=entry.get('key_env'),
                   max_input_tokens=entry.get('max_input_tokens')), entry
