"""Independent small standard-model checks of the near-unit readout candidate."""
from pathlib import Path
import argparse,datetime,json,sys,traceback
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compact_checkpoint import compile_compact,CompactDriver
from token_checkpoint import Q,RADIX
from compact_normalized_readout import apply_normalized_readout,normalize_emitted_digits,normalized_probability


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    report=dict(status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=[])
    def save():(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        m,l=compile_compact(1);d=CompactDriver(m,l);program=['H 0','T 0','H 0','X 0','T 0']
        old_trace=d.run_program(program);old_state=d.diagnostic_state().clone()
        report['patch']=apply_normalized_readout(m,l)
        new_trace=d.run_program(program);report['gate_trace_unchanged']=old_trace==new_trace
        report['gate_state_unchanged']=torch.equal(old_state,d.diagnostic_state());save()
        for event,total in [(0.,1.),(.125,.9999),(.5,.9999),(.9999,.9999),(.5,1.0001),(1.0001,1.0001)]:
            e=divmod(round((event+1)*Q),RADIX);t=divmod(round((total+1)*Q),RADIX)
            eg=(e[0]*RADIX+e[1])/Q-1;tg=(t[0]*RADIX+t[1])/Q-1
            got=normalize_emitted_digits(d,e,t)
            row=dict(event=eg,total=tg,output=got['probability'],reference=eg/tg,error=abs(got['probability']-eg/tg))
            row['passed']=row['error']<2e-6;report['rows'].append(row);save();print(row,flush=True)
        d.run_program(['H 0','T 0','H 0']);before=d.diagnostic_state().clone()
        result=normalized_probability(d,[0,1]);expected=(1-2**-.5)/2
        report['phase']=dict(**result,reference=expected,error=abs(result['probability']-expected),state_unchanged=torch.equal(before,d.diagnostic_state()))
        report['status']='passed' if report['gate_trace_unchanged'] and report['gate_state_unchanged'] and all(r['passed'] for r in report['rows']) and report['phase']['error']<5e-5 and report['phase']['state_unchanged'] else 'failed_checks'
        (a.out/'calls.json').write_text(json.dumps(d.call_trace)+'\n')
    except Exception:report.update(status='failed_execution',traceback=traceback.format_exc());raise
    finally:report['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()


if __name__=='__main__':main()
