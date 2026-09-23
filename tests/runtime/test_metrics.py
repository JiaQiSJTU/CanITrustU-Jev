import unittest
from src.eval.final_metrics import group_summary, VERSIONS

class FinalMetricsTests(unittest.TestCase):
    def row(self, base, version, correct=True):
        return dict(base_id=base,id=base,version=version,status='ok',correct=correct,
                    gold='a',choice='a' if correct else 'b', probabilities={'a':.8 if correct else .2,'b':.2 if correct else .8},
                    canonical_gold='a',canonical_choice='a' if correct else 'b',canonical_labels=['a','b'],
                    elapsed_seconds=.1,group_id=base,label_preserving=True)
    def test_all_six_must_pass(self):
        rows=[self.row(b,v,not(b=='B' and v==VERSIONS[-1])) for b in ['A','B'] for v in VERSIONS]
        report=group_summary(rows)
        self.assertEqual(report['overall_accuracy'],11/12)
        self.assertEqual(report['all_versions_correct_accuracy'],.5)
        self.assertEqual(report['versions']['v0_original']['accuracy_all'],1.)
    def test_missing_version_stays_in_denominator(self):
        report=group_summary([self.row('A',v) for v in VERSIONS[:-1]])
        self.assertEqual(report['overall_accuracy'],5/6)
        self.assertEqual(report['all_versions_correct_accuracy'],0.)
        self.assertEqual(report['complete_six_version_samples'],0)
    def test_duplicate_version_rejected(self):
        with self.assertRaises(ValueError):group_summary([self.row('A',VERSIONS[0])]*2)
    def test_failed_call_cannot_pass_strict_score(self):
        rows=[self.row('A',v) for v in VERSIONS];rows[-1].update(status='error',correct=False)
        self.assertEqual(group_summary(rows)['all_versions_correct_accuracy'],0.)

if __name__=='__main__':unittest.main()
