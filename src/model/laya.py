"""Laya evaluation adapter."""
from .model_utils import CallFailure, finish, prefer_env, local_context
from src.dataset.schema import request
import os
import time



class Laya:
    def __init__(self, config):
        import laya
        max_len = config.get('max_len')
        if max_len is not None and (type(max_len) is not int or max_len <= 0):
            raise ValueError('Laya max_len must be a positive integer')
        self.model_id = prefer_env('LAYA_WEIGHTS', config.get('weights', 'convaiinnovations/laya'))
        self.agent = laya.load(self.model_id, device=config.get('device'), subfolder=prefer_env('LAYA_SUBFOLDER', config.get('subfolder')))
        self.context = local_context(self.agent.model.encoder.config, requested=max_len)
        # The stock Laya encoder reads this budget on each prediction.
        self.agent.cfg['max_len'] = self.context['max_input_tokens']
        self._log_device('loaded', self._device_info())

    def _device_info(self):
        model = getattr(self.agent, 'model', None)
        devices = sorted({str(p.device) for p in model.parameters()}) if hasattr(model, 'parameters') else []
        return {'agent_device': str(getattr(self.agent, 'device', 'unknown')),
                'parameter_devices': devices}

    def _log_device(self, stage, info):
        print(f'[laya] pid={os.getpid()} {stage}: agent_device={info["agent_device"]}, '
              f'parameter_devices={info["parameter_devices"]}', flush=True)

    def predict(self, record):
        body = request(record, self.model_id)
        before = self._device_info()
        start = time.perf_counter()
        out = self.agent.predict(body['state'], body['questions'])
        after = self._device_info()
        if not getattr(self, '_device_prediction_logged', False) or before != after:
            self._log_device('after prediction', after)
            self._device_prediction_logged = True
        if not isinstance(out, dict) or 'answers' not in out:
            raise CallFailure('Laya returned no answers', response=out)
        payload = {'model': out.get('model', self.model_id), **{k: v for k, v in out.items() if k != 'model'}}
        payload['backend_metadata'] = {
            **payload.get('backend_metadata', {}), 'pid': os.getpid(),
            'device_before': before, 'device_after': after,
            'max_len': getattr(self.agent, 'cfg', {}).get('max_len')}
        # Laya's native agent rounds each choice probability to four decimals.
        return finish(payload, record, start, probability_decimals=4)
