"""Evaluate the frozen six-version dataset without generating further perturbations."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

from tqdm import tqdm

from src.model.base import strict_json
from src.model.registry import load_model
from src.dataset.reader import inspect_input, iter_records
from .final_metrics import VERSIONS, final_report

def fsync_directory(path):
    fd=os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as e:
        if e.errno not in (errno.EINVAL, errno.ENOTSUP, getattr(errno, 'EOPNOTSUPP', errno.ENOTSUP)):
            raise
    finally:
        os.close(fd)

def attach_provider_return(row, pred=None, error=None):
    """Keep the provider body for later analysis, including unscorable returns."""
    if pred is not None:
        row.update(asdict(pred))
        if getattr(pred, 'truncation', None):
            row['truncation'] = pred.truncation
        if not pred.response_text and pred.raw:
            row['response_text']=json.dumps(strict_json(pred.raw), ensure_ascii=False)
        return
    if error is None:
        return
    text=getattr(error, 'response_text', None)
    parsed=getattr(error, 'response', None)
    if text:
        row['response_text']=text
    if parsed is not None:
        row['raw']=parsed
    status=getattr(error, 'response_status', None)
    if status is not None:
        row['response_status']=status
    headers=getattr(error, 'response_headers', None)
    if headers:
        row['response_headers']=headers

def append_prediction(dest, row):
    """Write one finished result and fsync it before the next request."""
    data=(json.dumps(strict_json(row), ensure_ascii=False, allow_nan=False)+'\n').encode()
    written=0
    while written<len(data):
        n=dest.write(data[written:])
        if not n:raise OSError('prediction write returned 0')
        written+=n
    os.fsync(dest.fileno())

def read_prediction_rows(path):
    """Load finished JSONL rows. A truncated final line is incomplete and dropped."""
    data=path.read_bytes()
    if not data:return []
    lines=data.splitlines();rows=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:row=json.loads(line)
        except json.JSONDecodeError:
            if i==len(lines)-1:continue
            raise
        if isinstance(row,dict):rows.append(row)
    return rows

def kept_ok(rows):
    """One successful return per base and version. Later ok rows replace earlier ones."""
    kept=[];index={}
    for row in rows:
        version,base=row.get('version'),row.get('base_id')
        if version not in VERSIONS or not base or row.get('status')!='ok':continue
        key=(base,version)
        if key in index:kept[index[key]]=row
        else:index[key]=len(kept);kept.append(row)
    return kept

def rewrite_predictions(path, rows):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb', buffering=0) as dest:
        for row in rows:append_prediction(dest, row)
        if not rows:os.fsync(dest.fileno())
    os.replace(tmp, path)

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
    p.add_argument('--resume',action='store_true',help='keep returned rows in --output and rerun failures and missing versions')
    args=p.parse_args()
    if args.limit_bases<0:p.error('limit must be nonnegative')
    config=json.loads(args.models_config.read_text())
    model,entry=load_model(args.model,config)
    output=args.output or Path('results')/('final-'+args.model+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True,exist_ok=True)
    pred_path=output/'predictions.jsonl'
    kept=[]
    if args.resume:
        if not pred_path.exists():p.error('Nothing to resume')
        kept=kept_ok(read_prediction_rows(pred_path))
        rewrite_predictions(pred_path, kept)
        print(f'resume kept {len(kept)} returned rows; rerunning failures and missing versions',flush=True)
    elif pred_path.exists():p.error('Output exists; choose a new directory or pass --resume')
    manifest,input_sha=inspect_input(args.input)
    run={'created_at':datetime.now(timezone.utc).isoformat(),'model':args.model,'model_config':entry,
         'mock':args.model=='mock','input':str(args.input),'input_sha256':input_sha,'python':platform.python_version(),
         'limit_bases':args.limit_bases,'expected_versions':VERSIONS,'cost':'not measured',
         'source_hashes':{str(f):file_hash(f) for f in sorted(Path('src').rglob('*.py'))}}
    if args.resume and (output/'run.json').exists():
        previous=json.loads((output/'run.json').read_text())
        if isinstance(previous,dict) and previous.get('created_at'):run['created_at']=previous['created_at']
        run['resumed_at']=datetime.now(timezone.utc).isoformat()
    run['resume_kept_ok']=len(kept)
    (output/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    done={(row['base_id'],row['version']):row for row in kept}
    results=[];seen=set();bases=set();ignored=set();counter=Counter();ok_n=err_n=0
    total=args.limit_bases*len(VERSIONS) if args.limit_bases else (manifest or {}).get('version_instances')
    if not total and args.input.is_file():
        with args.input.open('rb') as stream:total=sum(1 for line in stream if line.strip())
    with pred_path.open('ab', buffering=0) as dest, tqdm(total=total or None, desc='eval', unit='条', file=sys.stderr, dynamic_ncols=True) as bar:
        fsync_directory(output)
        for r in iter_records(args.input, manifest):
            base=r['base_id']
            if base in ignored:continue
            if base not in bases and args.limit_bases and len(bases)>=args.limit_bases:
                ignored.add(base);continue
            bases.add(base)
            key=(base,r['version'])
            if key in seen or r['version'] not in VERSIONS:raise ValueError('Duplicate/unknown final version')
            seen.add(key);counter[r['version']]+=1
            if key in done:
                results.append(done[key]);ok_n+=1
                bar.update(1);bar.set_postfix(ok=ok_n, err=err_n, refresh=False);continue
            ids={o['id']:o['canonical_id'] for o in r['options']}
            row={k:r[k] for k in ('base_id','version','dataset','scenario','decision_type','actual_operator','group_id')}
            row.update(id=base,record_id=r['id'],gold=r['gold'],canonical_gold=ids[r['gold']],
                       canonical_labels=list(ids.values()),label_preserving=True,option_count=len(ids))
            start=time.perf_counter()
            try:
                if len(ids)>entry.get('max_options',255):raise ValueError('Backend candidate limit exceeded; counts as failed evaluation')
                pred=model.predict(r)
                attach_provider_return(row, pred=pred)
                row.update(status='ok',correct=pred.choice==r['gold'],canonical_choice=ids[pred.choice])
            except Exception as e:
                row.update(status='error',correct=False,reason=type(e).__name__+': '+str(e),elapsed_seconds=time.perf_counter()-start)
                attach_provider_return(row, error=e)
            append_prediction(dest, row);results.append(row)
            ok_n+=row['status']=='ok';err_n+=row['status']!='ok'
            bar.update(1);bar.set_postfix(ok=ok_n, err=err_n, refresh=False)
    if not results:p.error('Empty evaluation input')
    if manifest and not args.limit_bases and len(bases)!=manifest['original_samples']:
        raise ValueError('Original sample count mismatch')
    report=final_report(results)
    (output/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    run.update(original_samples=len(bases),version_counts=dict(counter),errors=sum(r['status']=='error' for r in results))
    (output/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    print(json.dumps({'output':str(output),'originals':len(bases),'instances':len(results),'kept':len(kept),
                      'retried':len(results)-sum((row['base_id'],row['version']) in done for row in results),
                      'errors':run['errors'],'mock':run['mock']}))
    if run['errors']:raise SystemExit(1)

if __name__=='__main__':main()
