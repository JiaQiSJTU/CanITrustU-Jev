"""First-party Jev and explicitly configured System One-compatible endpoints."""
import json
import os
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from .base import parse_response
from src.dataset.schema import request

class Jev:
    def __init__(self, endpoint='https://api.typesafe.ai/v1/systemone', model='jev-1.13.0',
                 key_env='JEV_API_KEY', timeout=60, retries=2, **kwargs):
        self.endpoint = endpoint
        self.model = model
        self.api_key = os.getenv(key_env, '') if key_env else ''
        self.timeout = timeout
        self.retries = retries

    def make_payload(self, record):
        return request(record, self.model)

    def normalize_response(self, response):
        return response

    def predict(self, record):
        payload = self.make_payload(record)
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        wire = json.dumps(payload, ensure_ascii=False).encode()
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                with urlopen(Request(self.endpoint, data=wire, headers=headers, method='POST'), timeout=self.timeout) as resp:
                    response = json.loads(resp.read())
                response = self.normalize_response(response)
                if response['answers']['decision']['type'] != record.get('primitive', 'choice'):
                    raise ValueError('Response primitive does not match request')
                return parse_response(response, record['options'], time.perf_counter() - started)
            except HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504, 529) or attempt == self.retries:
                    # Do not persist HTTP bodies, which might echo auth or private configuration.
                    raise RuntimeError(f'HTTP {e.code} from decision endpoint') from None
                time.sleep(min(2 ** attempt, 8))


class SystemOneOpen(Jev):
    """Translate the source-verified /decide list protocol."""
    def make_payload(self, record):
        official = request(record, self.model)
        q = official['questions']['decision']
        q['id'] = 'decision'
        if q['type'] == 'choice':
            q['options'] = q.pop('criteria')
        return {'state': official['state'], 'questions': [q]}

    def normalize_response(self, response):
        answers = response['answers']
        if not isinstance(answers, list) or len(answers) != 1 or answers[0].get('id') != 'decision':
            raise ValueError('Expected one identified system-one-open answer')
        return {**response, 'answers': {'decision': answers[0]}}
