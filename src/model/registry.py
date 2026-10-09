"""Instantiate configured evaluation adapters."""
import os
from copy import deepcopy
from .mock import Mock
from .jev import Jev
from .system_one_open import SystemOneOpen

def effective_config(name, config):
    """Resolve non-secret environment overrides before recording run identity."""
    if name == 'mock':
        return {'backend': 'mock', 'purpose': 'pipeline test only'}
    entry = deepcopy(config['models'][name])
    if entry['backend'] in ('diffusion', 'sglang', 'python'):
        entry['adapter_contract'] = 'upstream-text-20260928-v1'
    if entry['backend'] == 'diffusion':
        from .diffusion import DEFAULT_CANVAS_LENGTH, DEFAULT_CANVAS_STEP
        entry.setdefault('canvas_length', DEFAULT_CANVAS_LENGTH)
        entry.setdefault('canvas_step', DEFAULT_CANVAS_STEP)
        entry.setdefault('upstream_method', 'openjev' if name == 'openjev' else 'djev')
        entry.setdefault('confidence_method', 'normalized_entropy' if name == 'openjev' else 'max_probability')
    if entry['backend'] == 'sglang':
        from .openjev_sglang import ENGINE_DEFAULTS
        entry['engine_kwargs'] = {**ENGINE_DEFAULTS, **entry.get('engine_kwargs', {})}
    if entry['backend'] == 'python':
        entry.setdefault('mode', 'packed')
    overrides = {
        'semif': {'SEMIF_MODEL': 'request_model', 'SEMIF_REVISION': 'weights_revision'},
        'open_alternative_jev': {'SO1_WEIGHTS': 'weights'},
        'jeff': {'JEFF_BASE': 'base_model', 'JEFF_WEIGHTS': 'weights'},
        'kev_0_6b': {'KEV_BASE': 'base', 'KEV_WEIGHTS': 'weights'},
        'laya': {'LAYA_WEIGHTS': 'weights', 'LAYA_SUBFOLDER': 'subfolder'},
    }
    mapping = dict(overrides.get(name, {}))
    if entry.get('weights_env'):
        mapping[entry['weights_env']] = 'weights'
    for env, field in mapping.items():
        if os.environ.get(env) or (env == 'SEMIF_REVISION' and env in os.environ):
            entry[field] = os.environ[env] or None
    if entry.get('endpoint_env'):
        entry['default_endpoint'] = os.getenv(entry['endpoint_env'], entry.get('default_endpoint', ''))
    for field in ('weights', 'base', 'base_model', 'request_model'):
        value = entry.get(field)
        if value and os.path.isdir(value):
            if field == 'base' and name == 'kev_0_6b':
                from .model_utils import resolve_hub_snapshot
                value = resolve_hub_snapshot(value)
            entry[field] = os.path.realpath(value)
    if name == 'open_alternative_jev' and os.path.isdir(entry['weights']):
        entry.setdefault('load_kwargs', {}).pop('revision', None)
    return entry

def load_model(name, config):
    if name == 'mock':
        return Mock(), {'backend': 'mock', 'purpose': 'pipeline test only'}
    entry = effective_config(name, config)
    if entry['backend'] == 'djev_upstream':
        from .djev import Djev
        return Djev(entry), entry
    if entry['backend'] == 'openjev_upstream':
        from .openjev import OpenJev
        return OpenJev(entry), entry
    if entry['backend'] == 'sglang_upstream':
        from .openjev_sglang import OpenJevSglangService
        return OpenJevSglangService(entry), entry
    if entry['backend'] == 'python':
        from src.model.open_alternative_jev import LocalLibrary
        return LocalLibrary(entry), entry
    if entry['backend'] == 'semif':
        from src.model.semif import SemIf
        return SemIf(entry), entry
    if entry['backend'] == 'bridge':
        from src.model.bridge import Bridge
        return Bridge(entry), entry
    if entry['backend'] == 'laya':
        from src.model.laya import Laya
        return Laya(entry), entry
    if entry['backend'] == 'jeff':
        from src.model.jeff import Jeff
        return Jeff(entry), entry
    if entry['backend'] == 'kev':
        from src.model.kev import Kev
        return Kev(entry), entry
    if entry['backend'] == 'diffusion':
        from src.model.diffusion import DiffusionLocal
        return DiffusionLocal(entry), entry
    if entry['backend'] == 'sglang':
        from src.model.openjev_sglang import OpenJevSglang
        return OpenJevSglang(entry), entry
    endpoint = os.getenv(entry['endpoint_env'], entry.get('default_endpoint', ''))
    if not endpoint:
        raise ValueError('Set endpoint environment variable ' + entry['endpoint_env'])
    adapter = SystemOneOpen if entry['backend'] == 'system_one_http' else Jev
    return adapter(endpoint=endpoint, model=entry['request_model'], key_env=entry.get('key_env'),
                   max_input_tokens=entry.get('max_input_tokens'),
                   probability_decimals=entry.get('probability_decimals', 4)), entry
