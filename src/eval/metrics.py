import math
import random
from collections import defaultdict

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def quantile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)]

def summarize(rows):
    good = [r for r in rows if r['status'] == 'ok']
    applicable = [r for r in rows if r['status'] != 'not_applicable']
    result = {'scheduled': len(rows), 'not_applicable': len(rows) - len(applicable),
              'attempted': len(applicable), 'valid': len(good), 'errors': len(applicable) - len(good),
              'coverage': len(good) / len(applicable) if applicable else None,
              'accuracy_all': sum(r.get('correct', False) for r in applicable) / len(applicable) if applicable else None}
    if not good:
        return result
    result['accuracy_valid'] = mean([int(r['correct']) for r in good])
    result['nll'] = mean([-math.log(max(r['probabilities'][r['gold']], 1e-15)) for r in good])
    result['brier'] = mean([sum((v - int(k == r['gold'])) ** 2 for k, v in r['probabilities'].items()) for r in good])
    # Confidence here means probability assigned to the selected class, not entropy-based provider confidence.
    bins = defaultdict(list)
    for r in good:
        p = r['probabilities'][r['choice']]
        bins[min(9, int(p * 10))].append((p, int(r['correct'])))
    result['ece_10_bins'] = sum(len(v) / len(good) * abs(mean([p for p, _ in v]) - mean([y for _, y in v])) for v in bins.values())
    label_sets = {tuple(sorted(r.get('canonical_labels', []))) for r in good}
    if len(label_sets) == 1 and next(iter(label_sets)):
        labels = next(iter(label_sets))
        f1s, recalls = [], []
        for label in labels:
            tp = sum(r['canonical_gold'] == label and r['canonical_choice'] == label for r in good)
            fp = sum(r['canonical_gold'] != label and r['canonical_choice'] == label for r in good)
            fn = sum(r['canonical_gold'] == label and r['canonical_choice'] != label for r in good)
            if tp + fn:
                f1s.append(2 * tp / (2 * tp + fp + fn))
                recalls.append(tp / (tp + fn))
        result['macro_f1_present_gold_labels'] = mean(f1s)
        result['balanced_accuracy_present_gold_labels'] = mean(recalls)
    else:
        result['class_metrics_note'] = 'Omitted: instance-specific candidate sets have no common class semantics'
    latencies = [r['elapsed_seconds'] for r in good]
    result['latency_p50_seconds'] = quantile(latencies, .5)
    result['latency_p95_seconds'] = quantile(latencies, .95)
    result['usage_reporting_count'] = sum(bool(r.get('usage')) for r in good)
    result['total_input_tokens_reported'] = sum(r.get('usage', {}).get('input_tokens', 0) or 0 for r in good)
    result['total_output_tokens_reported'] = sum(r.get('usage', {}).get('output_tokens', 0) or 0 for r in good)
    return result

def paired(clean, perturbed, seed=42):
    base = {r['id']: r for r in clean if r['status'] == 'ok'}
    pairs = [(base[r['id']], r) for r in perturbed if r['status'] == 'ok' and r['id'] in base]
    if not pairs:
        return {'pairs': 0}
    deltas = [int(b['correct']) - int(a['correct']) for a, b in pairs]
    invariant = [(a, b) for a, b in pairs if b['label_preserving']]
    initial_correct = [(a, b) for a, b in invariant if a['correct']]
    grouped = defaultdict(list)
    for (a, b), d in zip(pairs, deltas):
        grouped[a['group_id']].append(d)
    groups = list(grouped.values())
    rng = random.Random(seed)
    boot = []
    for _ in range(500):
        sampled = [rng.choice(groups) for _ in groups]
        boot.append(mean([d for g in sampled for d in g]))
    return {'pairs': len(pairs), 'paired_accuracy_delta': mean(deltas),
            'paired_delta_group_bootstrap_95ci': [quantile(boot, .025), quantile(boot, .975)],
            'invariant_pairs': len(invariant),
            'canonical_flip_rate': mean([int(a['canonical_choice'] != b['canonical_choice']) for a, b in invariant]),
            'failure_given_clean_correct': mean([int(not b['correct']) for a, b in initial_correct]),
            'clean_correct_denominator': len(initial_correct)}

def report(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r['dataset'], r['variant'])].append(r)
    output = {}
    for (dataset, variant), group in sorted(by.items()):
        item = summarize(group)
        if variant != 'clean':
            item['paired'] = paired(by.get((dataset, 'clean'), []), group)
        output.setdefault(dataset, {})[variant] = item
    return output
