"""Explicit integration hook for native services without a verified Jev HTTP protocol.
Command receives one Jev request on stdin and returns one Jev response on stdout.
No shell is used. Startup belongs to reported latency. See the root README.
"""
import json
import os
import subprocess
import time
from .base import CallFailure, parse_json_or_none, parse_response
from src.dataset.schema import request

class Bridge:
    def __init__(self, config):
        raw = os.environ.get(config['command_env'])
        if not raw:
            raise ValueError('Configure JSON argv in ' + config['command_env'] + '; native bridge not supplied by upstream')
        self.argv = json.loads(raw)
        if not isinstance(self.argv, list) or not self.argv or not all(isinstance(x, str) for x in self.argv):
            raise ValueError('Bridge command must be a nonempty JSON array of strings')
        self.model = config['request_model']
        self.timeout = config.get('timeout', 180)

    def predict(self, record):
        start = time.perf_counter()
        proc = subprocess.run(self.argv, input=json.dumps(request(record, self.model)), text=True,
                              capture_output=True, timeout=self.timeout, check=False)
        text = proc.stdout or ''
        parsed = parse_json_or_none(text)
        if proc.returncode:
            raise CallFailure('Local bridge failed with exit code ' + str(proc.returncode), text, parsed)
        try:
            if not isinstance(parsed, dict):
                raise ValueError('Bridge stdout was not a JSON object')
            pred = parse_response(parsed, record['options'], time.perf_counter() - start)
        except CallFailure:
            raise
        except Exception as e:
            raise CallFailure(str(e), text, parsed) from None
        pred.response_text = text
        pred.raw = parsed
        return pred
