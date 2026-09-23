"""Evaluate the frozen six-version dataset without generating further perturbations."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

from src.model.registry import load_model
from src.dataset.reader import inspect_input, iter_records
from .final_metrics import VERSIONS, final_report

def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=Path('data/release'))
    p.add_argument('--model',default='mock')
    p.add_argument('--models-config',type=Path,default=Path('configs/models.json'))
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--limit-bases',type=int,default=0,help='0 = all 2000 originals and 12000 versions')
    args=p.parse_args()
    if args.limit_bases<0:p.error('limit must be nonnegative')
    config=json.loads(args.models_config.read_text())
    model,entry=load_model(args.model,config)
    output=args.output or Path('results')/('final-'+args.model+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True,exist_ok=True)
    if (output/'predictions.jsonl').exists():p.error('Output exists; choose a new directory')
    manifest,input_sha=inspect_input(args.input)
    run={'created_at':datetime.now(timezone.utc).isoformat(),'model':args.model,'model_config':entry,
         'mock':args.model=='mock','input':str(args.input),'input_sha256':input_sha,'python':platform.python_version(),
         'limit_bases':args.limit_bases,'expected_versions':VERSIONS,'cost':'not measured',
         'source_hashes':{str(f):file_hash(f) for f in sorted(Path('src').rglob('*.py'))}}
    (output/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    results=[];seen=set();bases=set();ignored=set();counter=Counter()
    with (output/'predictions.jsonl').open('w') as dest:
        for r in iter_records(args.input, manifest):
            base=r['base_id']
            if base in ignored:continue
            if base not in bases and args.limit_bases and len(bases)>=args.limit_bases:
                ignored.add(base);continue
            bases.add(base)
            key=(base,r['version'])
            if key in seen or r['version'] not in VERSIONS:raise ValueError('Duplicate/unknown final version')
            seen.add(key);counter[r['version']]+=1
            ids={o['id']:o['canonical_id'] for o in r['options']}
            row={k:r[k] for k in ('base_id','version','dataset','scenario','decision_type','actual_operator','group_id')}
            row.update(id=base,record_id=r['id'],gold=r['gold'],canonical_gold=ids[r['gold']],
                       canonical_labels=list(ids.values()),label_preserving=True,option_count=len(ids))
            start=time.perf_counter()
            try:
                if len(ids)>entry.get('max_options',255):raise ValueError('Backend candidate limit exceeded; counts as failed evaluation')
                pred=model.predict(r);row.update(asdict(pred))
                row.update(status='ok',correct=pred.choice==r['gold'],canonical_choice=ids[pred.choice])
            except Exception as e:
                row.update(status='error',correct=False,reason=type(e).__name__+': '+str(e),elapsed_seconds=time.perf_counter()-start)
            dest.write(json.dumps(row,ensure_ascii=False)+'\n');dest.flush();results.append(row)
            if len(results)%1000==0:print(f'evaluated {len(results)} instances',flush=True)
    if not results:p.error('Empty evaluation input')
    if manifest and not args.limit_bases and len(bases)!=manifest['original_samples']:
        raise ValueError('Original sample count mismatch')
    report=final_report(results)
    (output/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    run.update(original_samples=len(bases),version_counts=dict(counter),errors=sum(r['status']=='error' for r in results))
    (output/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    print(json.dumps({'output':str(output),'originals':len(bases),'instances':len(results),'errors':run['errors'],'mock':run['mock']}))
    if run['errors']:raise SystemExit(1)

if __name__=='__main__':main()
