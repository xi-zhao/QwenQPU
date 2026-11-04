"""Reproduce four final r5 figures from the verified 96-condition data extract."""
from pathlib import Path
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'review/cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'review/cache'))
sys.path.insert(0,str(ROOT/'research/standard_layers/benchmark_native_r5'))
from score import summarize


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    datafile=ROOT/'evidence/added-layers/r5-plot-data.json'
    tasksfile=ROOT/'evidence/added-layers/r5-tasks.json'
    assert sha(datafile)=='051643542579d3c78719c4db2fd6a3d777d5d447e6b8acccbf6ae2e94c52b9f6'
    assert sha(tasksfile)=='b6d892f5441f7b66fcc63d3c4a194188b5b84c410376225ae9d779feb01d0759'
    data=json.loads(datafile.read_text());tasks=json.loads(tasksfile.read_text())
    lookup={t['id']:t for t in tasks}
    rows=[dict(zip(data['columns'],r)) for r in data['rows']]
    assert len(rows)==96 and len({(r['id'],r['condition']) for r in rows})==96
    for r in rows:
        t=lookup[r['id']]
        r.update({k:t[k] for k in ('family','qubits','gate_count')})
        if r['absolute_errors'] is None:r['absolute_errors']={}
        assert r['total_predictions']==r['language_tokens']+r['native_predictions']<=16384
    groups={'nonclifford_primary':[t['id'] for t in tasks if t['family']=='nonclifford'],
            'clifford_control':[t['id'] for t in tasks if t['family']=='clifford'],
            **{f'qubits_{n}':[t['id'] for t in tasks if t['qubits']==n] for n in [3,4,6]},
            **{f'nonclifford_gates_{g}':[t['id'] for t in tasks if t['family']=='nonclifford' and t['gate_count']==g] for g in [12,18,24]}}
    result=dict(protocol='native-added-layers-computation-r5-formal',complete=True,rows=rows,
                **summarize(rows),subsets={k:summarize([r for r in rows if r['id'] in ids]) for k,ids in groups.items()},
                source_score_sha256=data['score_sha256'],data_extract_sha256=sha(datafile),
                purpose='Plotting extract, not a replacement for the complete independently reviewed score')
    assert [result['summary'][c]['correct'] for c in ['plain','qwenqpu']]==[4,48]
    for c,language,native,seconds in [('plain',238639,0,11138.899057726143),('qwenqpu',8007,543120,24974.0916750459)]:
        rr=[r for r in rows if r['condition']==c]
        assert sum(r['language_tokens'] for r in rr)==language
        assert sum(r['native_predictions'] for r in rr)==native
        assert abs(sum(r['seconds'] for r in rr)-seconds)<1e-8
    p=ROOT/'evidence/added-layers/r5-plot-score.json';p.write_text(json.dumps(result,indent=2)+'\n')
    table=[]
    for row in rows:
        r=dict(row);errors=r.pop('absolute_errors')
        r['error_A']=errors.get('A');r['error_B']=errors.get('B');table.append(r)
    with (ROOT/'evidence/added-layers/r5-per-condition.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    out=ROOT/'assets/r5';out.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(ROOT/'tools/plot_r5.py'),'--score',str(p),'--out',str(out)],check=True)
    meta=json.loads((out/'figure-metadata.json').read_text())
    meta.update(source_score_sha256=data['score_sha256'],data_extract_sha256=sha(datafile),
                task_sha256=sha(tasksfile),reproduction_command='python tools/render_r5_results.py')
    (out/'figure-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    summary={'version':result['protocol'],'status':'completed','completed_utc':'2026-11-04T04:46:32.378765+00:00',
             'conditions':96,'tasks':48,'score_sha256':data['score_sha256'],
             'frozen_sha256':'37f03134726b33a8c78785dfd1f4e41a067e71560fcc3c70f36ef624f3f00421',
             'summary':result['summary'],'paired':result['paired'],'subsets':result['subsets'],
             'max_qpu_probability_error':max(e for r in rows if r['condition']=='qwenqpu' for e in r['absolute_errors'].values()),
             'failures':{'plain':{'probability_wrong':41,'conflicting_answers':2,'missing_required_probabilities':1},'qwenqpu':{}},
             'termination':{'plain':{'eos':48},'qwenqpu':{'tool_call_then_numeric_completion':48}},
             'model_alive':False,'gpu_after':'1 MiB used, 81038 MiB free, utilization 0%',
             'integrity_review':{'conditions':96,'frozen_files':27,'sha256':'48b158248e43df3942250c927a904180f06d53ce88df2904915c384d4339d370'},
             'reviewed_archive':{'name':'QwenQPU-native-benchmark-r5-reviewed.zip','bytes':5269820,'sha256':'58870ed526b9a9d0191607df4d7075688e0eee550278942143a7352d578f93bc'}}
    summary['cost']={c:dict(language_tokens=sum(r['language_tokens'] for r in rows if r['condition']==c),
        native_predictions=sum(r['native_predictions'] for r in rows if r['condition']==c),
        total_seconds=sum(r['seconds'] for r in rows if r['condition']==c),
        median_seconds=statistics.median(r['seconds'] for r in rows if r['condition']==c)) for c in ['plain','qwenqpu']}
    (ROOT/'evidence/added-layers/r5-final.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['summary']))


if __name__=='__main__':main()
