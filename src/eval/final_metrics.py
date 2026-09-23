"""Six-version evaluation: sample, version, and all-versions-correct denominators."""
from collections import defaultdict
from .metrics import summarize, paired

VERSIONS = ['v0_original','v1_order','v2_label','v3_format','v4_context','v5_paraphrase']

def group_summary(rows, expected_versions=VERSIONS):
    by_version=defaultdict(list);by_base=defaultdict(list)
    for row in rows:
        if row['version'] not in expected_versions:
            raise ValueError('Unknown version: ' + row['version'])
        by_version[row['version']].append(row);by_base[row['base_id']].append(row)
    expected=set(expected_versions);strict=0;complete=0
    for base, group in by_base.items():
        versions=[r['version'] for r in group]
        if len(versions)!=len(set(versions)):
            raise ValueError('Duplicate version for base '+base)
        covered=set(versions)==expected
        complete+=int(covered)
        strict+=int(covered and all(r['status']=='ok' and r['correct'] for r in group))
    n=len(by_base);instances=n*len(expected)
    correct=sum(r['status']=='ok' and r.get('correct',False) for r in rows)
    output={'original_samples':n,'expected_version_instances':instances,'observed_instances':len(rows),
            'complete_six_version_samples':complete,'correct_instances':correct,
            'overall_accuracy':correct/instances if instances else None,
            'all_versions_correct_samples':strict,'all_versions_correct_accuracy':strict/n if n else None,
            'versions':{v:summarize(by_version.get(v,[])) for v in expected_versions}}
    for v in expected_versions:
        vr=by_version.get(v,[])
        item=output['versions'][v]
        item['expected']=n
        item['missing']=n-len(vr)
        item['accuracy_all']=sum(r['status']=='ok' and r.get('correct',False) for r in vr)/n if n else None
        item['coverage']=sum(r['status']=='ok' for r in vr)/n if n else None
        if v!='v0_original':
            output['versions'][v]['paired_vs_original']=paired(by_version.get('v0_original',[]),by_version.get(v,[]))
    return output

def final_report(rows):
    report=group_summary(rows)
    for key in ('dataset','scenario','decision_type'):
        groups=defaultdict(list)
        for row in rows:groups[row[key]].append(row)
        report['by_'+key]={k:group_summary(v) for k,v in sorted(groups.items())}
    operators=defaultdict(list)
    for row in rows:operators[row['actual_operator']].append(row)
    report['by_actual_operator']={k:summarize(v) for k,v in sorted(operators.items())}
    report['definition']='overall = correct / (originals * 6); strict = originals with all six valid and correct / originals; missing or failed versions cannot pass'
    return report
