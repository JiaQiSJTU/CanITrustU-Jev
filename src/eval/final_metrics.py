"""Score complete six-version questions, with provider-confidence abstention."""
from collections import defaultdict
import math
from .metrics import summarize, paired

VERSIONS = ['v0_original','v1_order','v2_label','v3_format','v4_context','v5_paraphrase']
CONFIDENCE_THRESHOLD = 0.5


def has_confidence(row):
    value = row.get('confidence')
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def group_summary(rows, expected_versions=VERSIONS):
    by_base = defaultdict(list)
    for row in rows:
        if row['version'] not in expected_versions:
            raise ValueError('Unknown version: ' + row['version'])
        by_base[row['base_id']].append(row)
    included = []
    complete = 0
    for base, group in by_base.items():
        versions = [r['version'] for r in group]
        if len(versions) != len(set(versions)):
            raise ValueError('Duplicate version for base ' + base)
        covered = set(versions) == set(expected_versions)
        complete += int(covered)
        if covered and all(r['status'] == 'ok' for r in group):
            included.append(group)
    good = [r for group in included for r in group]
    n, total = len(included), len(by_base)
    correct = sum(r['correct'] for r in good)
    strict = sum(all(r['correct'] for r in group) for group in included)
    confidence_count = sum(has_confidence(r) for r in good)
    confidence_complete = bool(good) and confidence_count == len(good)
    def abstains(r):
        return has_confidence(r) and r['confidence'] < CONFIDENCE_THRESHOLD
    def passes(r):
        return r['correct'] or abstains(r)
    output = {
        'original_samples': total, 'included_questions': n, 'excluded_questions': total-n,
        'expected_version_instances': total*len(expected_versions), 'observed_instances': len(rows),
        'complete_six_version_samples': complete, 'included_version_instances': len(good),
        'failed_requests': sum(r['status'] != 'ok' for r in rows),
        'correct_instances': correct, 'overall_accuracy': correct/len(good) if good else None,
        'mean_accuracy': correct/len(good) if good else None,
        'all_versions_correct_samples': strict,
        'all_versions_correct_accuracy': strict/n if n else None,
        'strict_accuracy': strict/n if n else None,
        'confidence_threshold': CONFIDENCE_THRESHOLD, 'confidence_reporting_count': confidence_count,
        'abstention_rate': sum(abstains(r) for r in good)/len(good) if confidence_complete else None,
        'mean_abstention_aware_accuracy': sum(passes(r) for r in good)/len(good) if confidence_complete else None,
        'strict_abstention_aware_accuracy': sum(all(passes(r) for r in g) for g in included)/n if confidence_complete else None,
        'versions': {},
    }
    if good and not confidence_complete:
        output['confidence_note'] = 'Abstention metrics unavailable: some included decisions lack provider confidence; class probabilities are not substituted.'
    by_version = {v: [r for r in good if r['version'] == v] for v in expected_versions}
    for v, vr in by_version.items():
        item = summarize(vr)
        item.update(expected=n, missing=0)
        if v != 'v0_original':
            item['paired_vs_original'] = paired(by_version.get('v0_original', []), vr)
        output['versions'][v] = item
    return output


def answer_format(row):
    if row['dataset'] in ('bfcl', 'mind2web', 'weblinx'):
        return 'dynamic_candidate_selection'
    if row['dataset'] in ('agentprocess', 'injecagent'):
        return 'fixed_category_classification'
    return 'yes_no_judgment'


def final_report(rows):
    report = group_summary(rows)
    for key in ('dataset', 'scenario', 'decision_type', 'answer_format'):
        groups = defaultdict(list)
        for row in rows:
            groups[answer_format(row) if key == 'answer_format' else row[key]].append(row)
        report['by_'+key] = {k: group_summary(v) for k, v in sorted(groups.items())}
    # Operator views use the same complete-question cohort as the main scores.
    groups = defaultdict(list)
    for row in rows:
        groups[row['base_id']].append(row)
    operators = defaultdict(list)
    for group in groups.values():
        if {r['version'] for r in group} == set(VERSIONS) and all(r['status'] == 'ok' for r in group):
            for row in group:
                operators[row['actual_operator']].append(row)
    report['by_actual_operator'] = {k: summarize(v) for k, v in sorted(operators.items())}
    report['definition'] = 'Exclude an entire question if any of its six versions is missing or failed. Mean averages decisions; strict requires all six to pass. Abstention-aware passes correct answers or provider confidence < 0.5; unavailable without complete provider confidence.'
    return report
