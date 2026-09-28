"""First-party Jev HTTP evaluation adapter."""
import json
import os
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from .model_utils import CallFailure, parse_response, fit_record, _header_map, _decode_return
from src.dataset.schema import request


class Jev:
    def __init__(self, endpoint='https://api.typesafe.ai/v1/systemone', model='jev-1.13.0',
                 key_env='JEV_API_KEY', timeout=60, retries=2, max_input_tokens=None, **kwargs):
        self.endpoint = endpoint
        self.model = model
        self.api_key = os.getenv(key_env, '') if key_env else ''
        self.timeout = timeout
        self.retries = retries
        self.max_input_tokens = max_input_tokens

    def make_payload(self, record):
        return request(record, self.model)

    def normalize_response(self, response):
        return response

    def predict(self, record):
        truncation = None
        if self.max_input_tokens:
            record, truncation = fit_record(record, self.max_input_tokens, self.make_payload)
        payload = self.make_payload(record)
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        wire = json.dumps(payload, ensure_ascii=False).encode()
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                with urlopen(Request(self.endpoint, data=wire, headers=headers, method='POST'), timeout=self.timeout) as resp:
                    status = getattr(resp, 'status', None)
                    response_headers = _header_map(getattr(resp, 'headers', None), self.api_key)
                    text, response = _decode_return(resp.read(), self.api_key)
                try:
                    normalized = self.normalize_response(response)
                    if not isinstance(normalized, dict):
                        raise ValueError('Response JSON must be an object')
                    if normalized['answers']['decision']['type'] != record.get('primitive', 'choice'):
                        raise ValueError('Response primitive does not match request')
                    pred = parse_response(normalized, record['options'], time.perf_counter() - started)
                except CallFailure:
                    raise
                except Exception as e:
                    raise CallFailure(str(e), text, response, status, response_headers) from None
                pred.raw = response if isinstance(response, dict) else {'value': response}
                pred.response_text = text
                pred.response_status = status
                pred.response_headers = response_headers
                if truncation:
                    pred.truncation = truncation
                return pred
            except HTTPError as e:
                try:
                    body = e.read()
                except Exception:
                    body = b''
                text, response = _decode_return(body, self.api_key)
                response_headers = _header_map(getattr(e, 'hdrs', None) or getattr(e, 'headers', None), self.api_key)
                if e.code in (429, 500, 502, 503, 504, 529) and attempt < self.retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise CallFailure(f'HTTP {e.code} from decision endpoint', text, response, e.code, response_headers) from None
            except CallFailure:
                raise
