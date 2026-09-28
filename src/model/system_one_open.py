"""System One-compatible HTTP protocol adapter."""
from .jev import Jev
from src.dataset.schema import request


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
