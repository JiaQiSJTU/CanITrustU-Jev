"""Deterministic pipeline sanity check. This is NOT a Jev performance baseline."""
from .base import Prediction

class Mock:
    def predict(self, record):
        ids = [o['id'] for o in record['options']]
        return Prediction(ids[0], {k: 1 / len(ids) for k in ids}, served_model='mock-first-option')
