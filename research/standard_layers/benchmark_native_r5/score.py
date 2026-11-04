"""Independent Qiskit evaluation, separate from model execution."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent


def state(circuit):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    qc = QuantumCircuit(6)
    for instruction in circuit:
        gate, *positions = instruction.split()
        if gate not in ['H', 'X', 'T', 'CNOT']:
            raise ValueError('Invalid gate')
        getattr(qc, 'cx' if gate == 'CNOT' else gate.lower())(*map(int, positions))
    return Statevector.from_instruction(qc).data


def members(event):
    if not isinstance(event, dict) or any(k not in [str(i) for i in range(6)] or type(v) is not int or v not in (0, 1) for k, v in event.items()):
        raise ValueError('Expected qubit-indexed event')
    return [i for i in range(64) if all((i // 2**int(q)) % 2 == b for q, b in event.items())]


def wilson(k, n):
    z = 1.959963984540054
    d = 1 + z*z/n
    center = (k/n + z*z/(2*n))/d
    half = z * math.sqrt((k/n)*(1-k/n)/n + z*z/(4*n*n))/d
    return [max(0, center-half), min(1, center+half)]


def binom_cdf(k, n, p):
    return sum(math.comb(n, i)*p**i*(1-p)**(n-i) for i in range(k+1))


def cp_bounds(k, n, alpha=.025):
    # Each marginal is a 97.5% exact binomial interval; union bound gives
    # at least 95% coverage for the paired difference below.
    def solve(index, target):
        lo, hi = 0., 1.
        for _ in range(64):
            mid = (lo+hi)/2
            if binom_cdf(index, n, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2
    return [0. if k == 0 else solve(k-1, 1-alpha/2), 1. if k == n else solve(k, alpha/2)]


def main():
    import numpy as np
    import qiskit
    p = argparse.ArgumentParser()
    p.add_argument('--reference-out', type=Path)
    p.add_argument('--results', type=Path)
    p.add_argument('--freeze-dir', type=Path, default=HERE/'frozen')
    a = p.parse_args()
    for name, digest in json.loads((a.freeze_dir/'FROZEN.json').read_text()).items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest, name
    tasks = json.loads((a.freeze_dir/'tasks.json').read_text())
    protocol = json.loads((a.freeze_dir/'protocol.json').read_text())
    states = {t['id']:state(t['circuit']) for t in tasks}
    refs = {t['id']:{k:float(np.sum(np.abs(states[t['id']][members(v)])**2)) for k,v in t['events'].items()} for t in tasks}
    reference = dict(qiskit_version=qiskit.__version__, precision='complex128', probabilities=refs,
                     task_sha256=hashlib.sha256((a.freeze_dir/'tasks.json').read_bytes()).hexdigest())
    if a.reference_out:
        if a.reference_out.exists():
            assert json.loads(a.reference_out.read_text()) == reference
        else:
            a.reference_out.write_text(json.dumps(reference, indent=2)+'\n')
        print('REFERENCE_SHA256', hashlib.sha256(a.reference_out.read_bytes()).hexdigest())
    if a.results is None:
        return
    rows = []
    for t in tasks:
        for condition in ['plain', 'qwenqpu']:
            path = a.results/f'{t["id"]}-{condition}.json'
            row = dict(id=t['id'], family=t['family'], qubits=t['qubits'], gate_count=t['gate_count'], condition=condition,
                       passed=False, reference=refs[t['id']], status='not_run', failure='not_run')
            if not path.exists():
                rows.append(row); continue
            r = json.loads(path.read_text())
            row.update(status=r['status'], failure=r.get('error') or r.get('finish_reason'), finish_reason=r.get('finish_reason'),
                       seconds=r['seconds'], language_tokens=r['language_generated_tokens'], native_predictions=r['native_forward_predictions'],
                       total_predictions=r['total_predictions'], peak_allocated_bytes=r['peak_allocated_bytes'],
                       source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), parse_mode=r.get('parse_mode'))
            if r['status'] == 'answered':
                answer = r['answer']['probabilities']
                row['answer'] = answer
                errors = {k:abs(answer.get(k, float('inf'))-v) for k,v in refs[t['id']].items()}
                row['absolute_errors'] = {k:v if math.isfinite(v) else None for k,v in errors.items()}
                row['numeric_correct'] = set(answer)==set(refs[t['id']]) and all(math.isfinite(e) and e <= protocol['absolute_probability_tolerance'] for e in errors.values())
                row['request_faithful'] = True
                if condition == 'qwenqpu':
                    request = r['parsed']
                    try:
                        row['events_faithful'] = set(request['events'])==set(t['events']) and all(members(request['events'][k])==members(v) for k,v in t['events'].items())
                        overlap = float(abs(np.vdot(states[t['id']], state(request['circuit'])))**2)
                        row['circuit_infidelity'] = max(0., 1-overlap)
                        row['circuit_faithful'] = abs(1-overlap) < 1e-10
                        row['request_faithful'] = row['events_faithful'] and row['circuit_faithful']
                    except (ValueError, TypeError, KeyError):
                        row['request_faithful'] = False
                row['passed'] = row['numeric_correct'] and row['request_faithful']
                row['failure'] = None if row['passed'] else ('request_wrong' if not row['request_faithful'] else 'probability_wrong')
            rows.append(row)
    summary = {}
    for c in ['plain', 'qwenqpu']:
        k = sum(r['passed'] for r in rows if r['condition']==c)
        summary[c] = dict(correct=k, scheduled=len(tasks), wilson_95=wilson(k,len(tasks)))
    pairs = []
    for t in tasks:
        pair = {r['condition']:r['passed'] for r in rows if r['id']==t['id']}
        pairs.append(int(pair['qwenqpu'])-int(pair['plain']))
    plus, minus, n = pairs.count(1), pairs.count(-1), len(tasks)
    pl, ml = cp_bounds(plus,n), cp_bounds(minus,n)
    discordant = plus+minus
    paired = dict(qwenqpu_only=plus, plain_only=minus, ties=pairs.count(0), difference=(plus-minus)/n,
                  difference_exact_conservative_95=[pl[0]-ml[1], pl[1]-ml[0]],
                  interval_method='Bonferroni union of two 97.5% Clopper-Pearson marginal discordance intervals',
                  exact_mcnemar_two_sided=min(1., 2*binom_cdf(min(plus,minus),discordant,.5)) if discordant else 1.)
    result = dict(protocol=protocol['version'], complete=all(r['status']!='not_run' for r in rows), reference=reference,
                  rows=rows, summary=summary, paired=paired)
    result['subsets']={label:summarize([r for r in rows if r['id'] in selected])
                      for label,selected in {
                          'nonclifford_primary':[t['id'] for t in tasks if t['family']=='nonclifford'],
                          'clifford_control':[t['id'] for t in tasks if t['family']=='clifford'],
                          **{f'qubits_{n}':[t['id'] for t in tasks if t['qubits']==n] for n in [3,4,6]},
                          **{f'nonclifford_gates_{g}':[t['id'] for t in tasks if t['family']=='nonclifford' and t['gate_count']==g] for g in [12,18,24]}
                      }.items() if selected}
    (a.results/'external-score.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    fields = ['id','family','qubits','gate_count','condition','status','passed','failure','finish_reason','seconds','language_tokens','native_predictions','total_predictions','peak_allocated_bytes']
    with (a.results/'per-condition.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(complete=result['complete'], summary=summary, paired=paired), indent=2))


def summarize(rows):
    ids=list(dict.fromkeys(r['id'] for r in rows));n=len(ids)
    summary={c:dict(correct=sum(r['passed'] for r in rows if r['condition']==c),scheduled=n)
             for c in ['plain','qwenqpu']}
    for value in summary.values():value['wilson_95']=wilson(value['correct'],n)
    lookup={(r['id'],r['condition']):r['passed'] for r in rows}
    plus=sum(lookup[(i,'qwenqpu')] and not lookup[(i,'plain')] for i in ids)
    minus=sum(lookup[(i,'plain')] and not lookup[(i,'qwenqpu')] for i in ids)
    both=sum(lookup[(i,'plain')] and lookup[(i,'qwenqpu')] for i in ids)
    pl,ml=cp_bounds(plus,n),cp_bounds(minus,n)
    return dict(summary=summary,paired=dict(qwenqpu_only=plus,plain_only=minus,both_correct=both,
                both_failed=n-plus-minus-both,difference=(plus-minus)/n,
                difference_exact_conservative_95=[pl[0]-ml[1],pl[1]-ml[0]],
                exact_mcnemar_two_sided=min(1.,2*binom_cdf(min(plus,minus),plus+minus,.5)) if plus+minus else 1.))


if __name__ == '__main__':
    main()
