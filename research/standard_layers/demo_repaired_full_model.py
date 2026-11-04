"""Two unedited end-to-end probability demonstrations of the repaired prefix."""
from pathlib import Path
from types import SimpleNamespace
import argparse,datetime,gc,hashlib,json,sys,time,traceback
import numpy as np
import torch
from transformers import AutoTokenizer
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from compact_checkpoint import CompactLayout
from merge_language_checkpoint import build,FullCompactTokenDriver
from task_interface import execute_probability_task
from benchmark_io import parse_json
sys.path.insert(0,str(Path(__file__).parent/'experiments'))
from compact_normalized_readout import normalized_probability


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--prefix',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tasks=[dict(id='phase',circuit=['H 3','T 3','H 3'],events={'q3_one':{'3':1},'all_zero':{str(i):0 for i in range(6)}}),
           dict(id='GHZ',circuit=['H 0']+[f'CNOT {i} {i+1}' for i in range(5)],events={'all_zero':{str(i):0 for i in range(6)},'all_one':{str(i):1 for i in range(6)}})]
    system=json.loads((Path(__file__).parent/'benchmark-v6/protocol.json').read_text())['systems']['qwenqpu']
    protocol=dict(kind='Two development demonstrations; no new accuracy benchmark',tasks=tasks,system=system,
        thinking=False,decoding='greedy',language_limit=1024,total_prediction_limit=16384,
        probability_tolerance=5e-5,output='Model numeric probabilities; external correctness scoring',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        readout_adapter_sha256=hashlib.sha256((Path(__file__).parent/'experiments/compact_normalized_readout.py').read_bytes()).hexdigest())
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    report=dict(status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tasks=[])
    def save():
        p=a.out/'report.pending';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(a.out/'report.json')
    def cleanup():gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    def reference(program):
        q=QuantumCircuit(6)
        for op in program:
            parts=op.split();getattr(q,'cx' if parts[0]=='CNOT' else parts[0].lower())(*map(int,parts[1:]))
        return Statevector.from_instruction(q).data
    def mask(event):
        if not isinstance(event,dict) or any(k not in [str(i) for i in range(6)] or type(v) is not int or v not in [0,1] for k,v in event.items()):
            raise ValueError('Invalid independently scored event')
        return [int(all(format(i,'06b')[5-int(k)]==str(v) for k,v in event.items())) for i in range(64)]
    try:
        config=json.loads((a.prefix/'config.json').read_text());meta=config['qwenqpu_added_layer_prototype']
        assert meta['qubits']==6 and meta['normalized_readout'] and meta['sampling_revision']=='native-rejection-r7'
        report['prefix_SHA256']={x.name:hashlib.file_digest(x.open('rb'),'sha256').hexdigest() for x in a.prefix.glob('*.safetensors')}
        tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True,trust_remote_code=False)
        model,audit=build(a.base,a.prefix,'cuda',lambda key,i,total:print('BUILD',i,total,flush=True) if i%100==0 or i==total else None)
        report['build']=audit;save()
        for task in tasks:
            prompt='Six qubits start in |000000>. Apply this circuit in order: '+', '.join(task['circuit'])+'. Return the probabilities of these events: '+json.dumps(task['events'])+'.'
            messages=[dict(role='system',content=system),dict(role='user',content=prompt)]
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True)['input_ids'].to('cuda')
            cache=None;z=None;tokens=[];driver=None;torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            row=dict(id=task['id'],prompt=prompt,status='running',passed=False)
            try:
                for i in range(1024):
                    with torch.inference_mode():z=model(input_ids=ids if i==0 else torch.tensor([[tokens[-1]]],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1)
                    cache=z.past_key_values;tokens.append(int(z.logits[0,-1].argmax()))
                    if tokens[-1] in [248044,248046]:break
                raw=dict(messages=messages,prompt_ids=ids[0].tolist(),tokens=tokens,text=tokenizer.decode(tokens,skip_special_tokens=True),finish_reason='eos' if tokens[-1] in [248044,248046] else 'length')
                (a.out/f'{task["id"]}-language.json').write_text(json.dumps(raw,indent=2)+'\n')
                row.update(language=raw,language_tokens=len(tokens));cache=None;z=None;cleanup()
                if raw['finish_reason']!='eos':raise ValueError('Language budget exhausted')
                request=parse_json(raw['text']);driver=FullCompactTokenDriver(model,CompactLayout(6)).driver
                driver.prediction_budget=16384-len(tokens)
                original=driver._forward_step
                def bounded(tid,uniform=None):
                    if len(driver.call_trace)+len(tokens)>=16384:raise ValueError('Total prediction budget exhausted')
                    return original(tid,uniform)
                driver._forward_step=bounded
                transport=SimpleNamespace(lay=driver.lay,run_program=driver.run_program,
                    probability=lambda event:normalized_probability(driver,event))
                answer,execution=execute_probability_task(transport,request)
                (a.out/f'{task["id"]}-execution.json').write_text(json.dumps(dict(request=request,answer=answer,execution=execution,all_forward_calls=driver.call_trace))+'\n')
                expected=reference(task['circuit']);submitted=reference(request['circuit'])
                refs={k:float(np.dot(abs(expected)**2,mask(v))) for k,v in task['events'].items()}
                errors={k:abs(answer['probabilities'].get(k,float('inf'))-v) for k,v in refs.items()}
                faithful=set(request['events'])==set(task['events']) and all(mask(request['events'][k])==mask(v) for k,v in task['events'].items())
                overlap=float(abs(np.vdot(expected,submitted))**2)
                row.update(status='answered',request=request,answer=answer,reference=refs,errors=errors,
                    events_faithful=faithful,circuit_overlap=overlap,native_predictions=len(driver.call_trace),
                    passed=bool(max(errors.values())<5e-5 and faithful and abs(overlap-1)<1e-10))
            except (ValueError,KeyError,TypeError) as error:
                row.update(status='task_failed',error=str(error))
            except RuntimeError as error:
                row.update(status='execution_failure',error=str(error))
                report['tasks'].append(row);save();raise
            finally:
                torch.cuda.synchronize();row.update(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                cache=None;z=None
                if driver is not None:driver.cache=None;driver._forward_step=None
                driver=None;original=None;cleanup()
            report['tasks'].append(row);save();print('DEMO',task['id'],row['status'],row['passed'],flush=True)
        report['status']='passed' if all(r['passed'] for r in report['tasks']) else 'failed_checks'
        del model;cleanup();report['allocated_after_release_bytes']=torch.cuda.memory_allocated()
    except Exception:report.update(status='failed_execution',traceback=traceback.format_exc());raise
    finally:report['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()


if __name__=='__main__':main()
