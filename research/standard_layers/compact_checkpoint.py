"""Addressed digit RAM and shared gate arithmetic in stock Qwen layers.

One capacity-six checkpoint accepts arbitrary circuits on any of its six wires.
Three read ports share replicated 128-row banks; arithmetic has four forms,
independent of address. Circuit control and numeric writeback use native tokens.
The host loads the ROM and schedules event queries; it never updates amplitudes,
adds probabilities, or chooses a sampled outcome. This is a research compiler.
"""
from pathlib import Path
from dataclasses import dataclass
import collections
import hashlib
import json
import math
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from token_checkpoint import B, C, REF, SCALE, Q, RADIX, MASK, projection


@dataclass
class CompactLayout:
    n: int = 6

    def __post_init__(self):
        if type(self.n) is not int or not 1 <= self.n <= 6:
            raise ValueError('This compact layout has capacity 1..6 qubits')
        self.size=1<<self.n; self.components=2*self.size
        self.banks=2  # Fixed capacity: 256 component cells, two buffers.
        self.heads=16; self.values=32; self.hidden=1024
        self.state_heads=6; self.controller_head=6; self.rom_head=7
        self.acc_heads=(11,12)
        self.gates=[(g,q) for g in ['H','X','T'] for q in range(self.n)]
        self.gates += [('CNOT',c,t) for c in range(self.n) for t in range(self.n) if c!=t]
        self.init_gate=len(self.gates)
        # All coordinates have distinct owners. Four ballast coordinates pin
        # RMSNorm at the same scale as the established digit representation.
        self.g=8; self.p=9; self.src=10; self.ph=11; self.kind=12
        self.init_one=16; self.ctrl_flag=17; self.read_flag=18; self.load_flag=19
        self.qrow=20; self.qbank=404; self.wrow=410; self.wbank=538
        self.rom_row=540; self.rom_bank=668; self.rom_parity=672
        self.read=674; self.ctrl=680; self.pc=684; self.rom=685
        self.z=686; self.e=687; self.gray=688
        self.cg=700; self.cp=701; self.cs=702; self.cph=703
        self.ggray=704; self.pgray=710
        self.last_low=718; self.halt_flag=719; self.next_pc=720
        self.penalty=721; self.auto_penalty=723
        self.query_select=726; self.query_id=727; self.query_src=728
        self.prob_phase=729; self.event=731; self.acc_row=732; self.acc_write=734
        self.acc_reset=736; self.acc_read=737; self.acc_value=741
        self.r=742; self.im=743; self.born=744; self.prob_z=745
        self.sample_flag=746; self.uniform=747; self.sample_index=748
        self.sample_score=749; self.select_penalty=750; self.sample_penalty=751
        self.sample_onehot=752
        self.controls={}; tid=2*RADIX
        for g in range(len(self.gates)+1):
            for p in range(self.components):
                for source in range(2):
                    for phase in range(2):
                        self.controls[g,p,source,phase]=tid; tid+=1
        self.reset_id=tid; self.read_start=tid+1
        self.load_codes=len(self.gates)+1
        self.load_start=self.read_start+1024
        self.prob_reset=self.load_start+1024*self.load_codes
        self.query_select_start=self.prob_reset+1
        self.query_start=self.query_select_start+4*self.size
        self.sample_start=self.query_start+8*self.size
        self.outcome_start=self.sample_start+self.size
        self.continue_id=self.outcome_start+self.size
        self.halt_id=self.continue_id+1; self.resume_id=self.halt_id+1;self.vocab=self.resume_id+1
        self.autonomous=True; self.probability_readout=True
        path=Path(__file__).resolve().parents[2]/'weight-release/module-bundle/quantum_gates.pt'
        self.source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        if self.source_sha256!='6aac7d79842243f8e58a698fa01936458e0e79079fd099d3cd6e05613145f34c':
            raise ValueError('Fixed gate source changed')
        self.weights=torch.load(path,weights_only=True,map_location='cpu')['gates']

    def operation(self,g,p):
        """Compile addresses and one of four reusable arithmetic forms."""
        if g==self.init_gate:return 3,0,0
        op=self.gates[g]; a,ri=divmod(p,2); q=op[-1]
        lo=a&~(1<<q); hi=lo|(1<<q)
        if op[0]=='H':return (1 if not a>>q&1 else 2),2*lo+ri,2*hi+ri
        if op[0]=='T' and a>>q&1:return (2 if ri==0 else 1),2*a,2*a+1
        if op[0]=='X' or op[0]=='CNOT' and a>>op[1]&1:return 0,2*(a^(1<<q))+ri,0
        return 0,p,0

    def digit_form(self,port):
        return {self.read+2*port:RADIX/Q,self.read+2*port+1:1/Q,0:-1-C*(RADIX+1)/Q}


def compile_compact(capacity=6):
    l=CompactLayout(capacity)
    cfg=Qwen3_5TextConfig(vocab_size=l.vocab,hidden_size=l.hidden,intermediate_size=9216,
      num_hidden_layers=4,layer_types=['linear_attention']*4,
      num_attention_heads=2,num_key_value_heads=1,head_dim=64,
      linear_num_key_heads=16,linear_num_value_heads=32,
      linear_key_head_dim=128,linear_value_head_dim=128,tie_word_embeddings=False,
      attn_implementation='eager')
    cfg.qwenqpu_added_layer_prototype=dict(qubits=capacity,format='compact-addressed-v7',
      digit_radix=RADIX,amplitude_scale=Q,language_weights_included=False,
      autonomous_controller=True,probability_readout=True,
      source_quantum_gate_sha256=l.source_sha256,shared_arithmetic_forms=4,
      fixed_key_heads=16,host_schedules_event_queries=True,ballast_coordinates=[0,1,1022,1023])
    m=Qwen3_5ForCausalLM(cfg).float().eval().requires_grad_(False)
    for w in m.parameters():w.zero_()
    emb=m.model.embed_tokens.weight; emb[:,[0,1,1022,1023]]=B
    for phase in range(2):
        rows=emb[phase*RADIX:(phase+1)*RADIX]
        rows[:,2+phase]=1; rows[:,4]=torch.arange(RADIX)
    def address(row,port,cell):
        bank,r=divmod(cell,128);row[l.qrow+port*128+r]=1;row[l.qbank+port*2+bank]=1
    def write_address(row,cell):
        bank,r=divmod(cell,128);row[l.wrow+r]=1;row[l.wbank+bank]=1
    for (g,p,s,ph),tid in l.controls.items():
        row=emb[tid]; kind,a,b=l.operation(g,p)
        row[l.g]=g;row[l.p]=p;row[l.src]=s;row[l.ph]=ph
        row[l.kind+kind]=1;row[l.ctrl_flag]=1
        row[l.init_one]=int(g==l.init_gate and p==0)
        if g!=l.init_gate:
            address(row,0,s*l.components+a);address(row,1,s*l.components+b)
        address(row,2,(1-s)*l.components+p);write_address(row,(1-s)*l.components+p)
        row[5]=1 # gate arithmetic active
    row=emb[l.reset_id];row[l.ctrl_flag]=1;row[l.read_flag]=1
    row[l.g]=l.init_gate;row[l.src]=1;row[4]=-1
    emb[l.resume_id,l.read_flag]=1;emb[l.resume_id,4]=-1
    for pc in range(1024):
        for tid,code in [(l.read_start+pc,None)]+[(l.load_start+pc*l.load_codes+k,k) for k in range(l.load_codes)]:
            row=emb[tid];row[l.rom_row+(pc//2)%128]=1;row[l.rom_bank+pc//256]=1
            row[l.rom_parity+pc%2]=1
            if code is None:row[l.read_flag]=1;row[4]=pc
            else:row[l.load_flag]=1;row[4]=code
    emb[l.prob_reset,l.acc_reset]=1
    for a in range(l.size):
        for ph in range(2):
            for event in range(2):
                rid=4*a+2*ph+event
                emb[l.query_select_start+rid,l.query_select]=1
                emb[l.query_select_start+rid,l.query_id]=rid
                for s in range(2):
                    row=emb[l.query_start+2*rid+s]
                    address(row,0,s*l.components+2*a);address(row,1,s*l.components+2*a+1)
                    row[l.prob_phase+ph]=1;row[l.event]=event
                    row[l.acc_row+a%2]=1;row[l.acc_write+1-a%2]=1
        row=emb[l.sample_start+a];row[l.sample_flag]=1;row[l.sample_index]=a
        row[l.sample_onehot+a]=1
        row[l.acc_row+a%2]=1;row[l.acc_write+1-a%2]=1
    la=m.model.layers[0].linear_attn;kbase=l.heads*128;vbase=2*kbase
    def channel(index,form,tap=-1):
        projection(la.in_proj_qkv.weight[index],form);la.conv1d.weight[index,0,tap]=1
    def value(vh,forms,beta):
        for j,f in enumerate([{0:REF},{0:REF},*forms]):channel(vbase+vh*128+j,f)
        projection(la.in_proj_b.weight[vh],beta)
        la.in_proj_z.weight[vh*128:vh*128+128,0]=128/(B*SCALE)
    for port in range(3):
        for bank in range(2):
            h=2*port+bank
            for r in range(128):
                channel(h*128+r,{l.qrow+128*port+r:128,l.qbank+2*port+bank:128,0:-128})
                channel(kbase+h*128+r,{l.wrow+r:128,l.wbank+bank:128,0:-128},-2)
            for digit in range(2):
                vh=2*h+digit;value(vh,[{0:C,4:1}],{2+digit:256,0:-128})
                la.out_proj.weight[l.read+2*port+digit,vh*128+2]=REF/1024
    h=l.controller_head
    channel(h*128,{0:128});channel(kbase+h*128,{l.ctrl_flag:128,l.read_flag:128})
    value(2*h,[{0:C,d:1} for d in [l.g,l.p,l.src,l.ph]],{l.ctrl_flag:256,0:-128})
    value(2*h+1,[{0:C,4:1}],{l.read_flag:256,0:-128})
    for j in range(4):la.out_proj.weight[l.ctrl+j,2*h*128+2+j]=REF/1024
    la.out_proj.weight[l.pc,(2*h+1)*128+2]=REF/1024
    for bank in range(4):
        h=l.rom_head+bank
        for r in range(128):
            f={l.rom_row+r:128,l.rom_bank+bank:128,0:-128}
            channel(h*128+r,f);channel(kbase+h*128+r,f)
        for parity in range(2):
            vh=2*h+parity
            value(vh,[{0:C,4:1}],{l.load_flag:256,l.rom_parity+parity:256,0:-384})
            la.in_proj_z.weight[vh*128:vh*128+128].zero_()
            for j in range(128):projection(la.in_proj_z.weight[vh*128+j],{l.rom_parity+parity:128})
            la.out_proj.weight[l.rom,vh*128+2]=REF/1024
    # Two replicated accumulator ports read prior sum and destination high digit.
    for port,h in enumerate(l.acc_heads):
        for r in range(2):
            channel(h*128+r,{(l.acc_row if port==0 else l.acc_write)+r:128})
            channel(kbase+h*128+r,{l.acc_write+r:128},-2)
        # The reset uses a separate conv tap on the reset flag at row zero.
        projection(la.in_proj_qkv.weight[kbase+h*128],{l.acc_reset:128})
        la.conv1d.weight[kbase+h*128,0,-1]=1
        for digit in range(2):
            vh=2*h+digit
            value(vh,[{0:C,4:1,l.acc_reset:1024 if digit==0 else 0}],
                  {2+digit:256,l.acc_reset:256,0:-128})
            la.out_proj.weight[l.acc_read+2*port+digit,vh*128+2]=REF/1024
    la.dt_bias.fill_(-110);la.norm.weight.fill_(1)
    used=[]
    def builder(layer):
        mlp=m.model.layers[layer].mlp;counter=[0]
        def unit(gate,up,out,coef=1):
            i=counter[0];projection(mlp.gate_proj.weight[i],gate);projection(mlp.up_proj.weight[i],up)
            mlp.down_proj.weight[out,i]=coef;counter[0]+=1
        used.append(counter);return unit
    def mulform(form,factor):return {d:v*factor for d,v in form.items()}
    def addforms(*forms):
        out={}
        for f in forms:
            for d,v in f.items():out[d]=out.get(d,0)+v
        return out
    unit=builder(0)
    hcoef=float(l.weights[0,0,0]);x=l.digit_form(0);y=l.digit_form(1)
    forms=[x,addforms(mulform(x,hcoef),mulform(y,hcoef)),
           addforms(mulform(x,hcoef),mulform(y,-hcoef)),{l.init_one:1}]
    # 4 shared arithmetic forms x 2 phases, independent of gate/address count.
    for kind,e in enumerate(forms):
        for ph in range(2):
            pred={l.kind+kind:512,5:512,l.ph:512 if ph else -512,0:-1280 if ph else -768}
            z=mulform(e,Q/RADIX if ph==0 else Q)
            z[0]=z.get(0,0)+((Q+.5)/RADIX-.5 if ph==0 else Q+C*RADIX)
            if ph:z[l.read+4]=-RADIX
            unit(pred,mulform(z,1/256),l.z);unit(pred,mulform(e,1/256),l.e)
    for phase in range(2):
        unit({0:128},{0:-MASK/128,5:MASK/128,l.ph:(MASK/128 if phase else -MASK/128),
                      l.prob_phase+phase:MASK/128},l.penalty+phase)
    # Fix low-phase eligibility: gate-active AND phase, rather than its sum.
    # Replace the two penalty outputs with finite-state conjunctions below.
    m.model.layers[0].mlp.down_proj.weight[l.penalty:l.penalty+2].zero_()
    for ph in range(2):
        unit({0:128},{0:-MASK/128,l.prob_phase+ph:MASK/128},l.penalty+ph)
        unit({5:512,l.ph:512 if ph else -512,0:-768 if ph else -256},{0:MASK/256},l.penalty+ph)
    num={2:128,3:128}
    for out,f in [(l.cg,{l.ctrl:1,0:-C}),(l.cp,{l.ctrl+1:1,l.ctrl+3:1,0:-2*C}),
                  (l.cs,{l.ctrl+2:1,0:-C}),(l.cph,{l.ctrl+3:-1,0:C+1})]:
        unit(num,mulform(f,1/128),out)
    unit({l.read_flag:128},{l.rom:1/128,0:-C/128},l.cg)
    unit({l.read_flag:128},{l.ctrl+2:-1/128,0:(C+1)/128},l.cs)
    unit({0:128},{l.pc:1/128,0:(1-C)/128},l.next_pc)
    unit({l.query_select:128},{l.ctrl+2:-1/128,0:(C+1)/128},l.query_src)
    for port,out in [(0,l.r),(1,l.im)]:
        unit({l.prob_phase:512,l.prob_phase+1:512,l.event:512,0:-768},mulform(l.digit_form(port),1/256),out)
    unit({0:128},{l.acc_read:RADIX/Q/128,l.acc_read+1:1/Q/128,0:(-1-C*(RADIX+1)/Q)/128},l.acc_value)
    unit=builder(1)
    def gray(unit,src,dst,count,bits):
        for b in range(bits):
            seq=[((k^(k>>1))>>b)&1 for k in range(count)]
            slopes=[seq[k+1]-seq[k] for k in range(count-1)]
            changes=[slopes[0]]+[slopes[k]-slopes[k-1] for k in range(1,count-1)]+[-slopes[-1]]
            for knot,change in enumerate(changes):
                if change:unit({src:128,0:-128*knot},{0:1/128},dst+b,change)
    gray(unit,l.z,l.gray,RADIX,12)
    gray(unit,l.cg,l.ggray,64,6);gray(unit,l.cp,l.pgray,128,7)
    for d in [l.r,l.im]:
        for sign in [1,-1]:unit({d:sign},{d:1},l.born,sign)
    stride=l.components;last=stride-1
    f={l.ctrl+1:4,l.ctrl+3:4*stride,2:8*stride,3:8*stride,
       0:4*(-C*(1+stride)-last+.5-3*stride)}
    for shift,coef in [(1,1),(0,-1)]:
        z=dict(f);z[0]+=shift;unit(mulform(z,128),{0:1/128},l.last_low,coef)
    f={l.rom:4,l.read_flag:512,0:-4*(C+len(l.gates))-512}
    for shift,coef in [(2,1),(1,-1),(-1,-1),(-2,1)]:
        z=dict(f);z[0]+=shift;unit(mulform(z,128),{0:1/128},l.halt_flag,coef)
    unit=builder(2)
    for ph,factor in [(0,Q/RADIX),(1,Q)]:
        z={l.born:factor,l.acc_value:factor,0:factor}
        if ph:z.update({l.acc_read+2:-RADIX});z[0]+=C*RADIX
        else:z[0]+=.5/RADIX-.5
        unit({l.prob_phase+ph:128},mulform(z,1/128),l.prob_z)
    valid=[{2:1,3:1,l.last_low:-1,l.read_flag:1,l.halt_flag:-1},
           {l.last_low:1},{l.halt_flag:1}]
    for j,f in enumerate(valid):unit({0:128},addforms(mulform(f,MASK/128),{0:-MASK/128}),l.auto_penalty+j)
    unit({0:128},{l.query_select:MASK/128,0:-MASK/128},l.select_penalty)
    unit({0:128},{l.sample_flag:MASK/128,0:-MASK/128},l.sample_penalty)
    sharp=2**24
    current={l.acc_read+2:RADIX/Q,l.acc_read+3:1/Q,0:-1-C*(RADIX+1)/Q}
    for f in [{l.acc_value:sharp,l.uniform:-sharp},addforms(mulform(current,-sharp),{l.uniform:sharp})]:
        unit(f,{0:1/sharp},l.sample_score,-1)
    unit=builder(3);gray(unit,l.prob_z,l.gray,RADIX,12)
    assert max(x[0] for x in used)<=cfg.intermediate_size
    lm=m.lm_head.weight;lm[:,0]=-2*MASK/B
    def code(row,src,k,bits):
        g=k^(k>>1)
        for b in range(bits):row[src+b]=1 if g>>b&1 else -1
        row[0]-=g.bit_count()/B
    for ph in range(2):
        for k in range(RADIX):
            row=lm[ph*RADIX+k];row.zero_();code(row,l.gray,k,12);row[l.penalty+ph]=1
    for (g,p,s,ph),tid in l.controls.items():
        row=lm[tid];row.zero_();code(row,l.ggray,g,6);code(row,l.pgray,p,7)
        row[l.cs]=2*s-1;row[l.cph]=2*ph-1;row[0]-=(s+ph)/B
        row[l.auto_penalty]=1
    for pc in range(1024):
        row=lm[l.read_start+pc];row.zero_();row[l.next_pc]=2*pc;row[0]=-pc*pc/B;row[l.auto_penalty+1]=1
    lm[l.halt_id].zero_();lm[l.halt_id,l.auto_penalty+2]=1
    for rid in range(4*l.size):
        for s in range(2):
            row=lm[l.query_start+2*rid+s];row.zero_();row[l.query_id]=2*rid
            row[l.query_src]=2*s;row[0]=-(rid*rid+s)/B;row[l.select_penalty]=1
    for a in range(l.size):
        # All candidate outcomes and CONTINUE need the same score origin.
        # A quadratic address score would contribute a**2 to the selected
        # outcome, incorrectly overpowering the CDF interval test.
        row=lm[l.outcome_start+a];row.zero_();row[l.sample_onehot+a]=2
        row[0]=-2/B;row[l.sample_score]=1;row[l.sample_penalty]=1
    lm[l.continue_id].zero_();lm[l.continue_id,0]=-(2**-25)/B
    lm[l.continue_id,l.sample_penalty]=1
    cfg.qwenqpu_added_layer_prototype['MLP_units_used']=[x[0] for x in used]
    return m,l


class CompactDriver:
    def __init__(self,model,layout):
        self.model=model;self.lay=layout;self.cache=None;self.trace=[];self.call_trace=[]
    def _forward_step(self,tid,uniform=None):
        device=next(self.model.parameters()).device;ids=torch.tensor([[tid]],device=device)
        # The stock HF cache derives default positions from a full-attention
        # layer. This all-GDN prefix has none. Explicit positions/mask avoid
        # that inference; GDN itself does not use rotary positions.
        kw={'input_ids':ids,'position_ids':torch.zeros_like(ids),
            'attention_mask':{'linear_attention':None,'full_attention':None}}
        if uniform is not None:
            x=self.model.model.embed_tokens(ids).clone();x[0,0,self.lay.uniform]=uniform
            del kw['input_ids'];kw['inputs_embeds']=x
        with torch.inference_mode():out=self.model(**kw,past_key_values=self.cache,use_cache=True,logits_to_keep=1)
        self.cache=out.past_key_values;return int(out.logits[0,-1].argmax())
    def step(self,tid,uniform=None):
        result=self._forward_step(tid,uniform)
        self.call_trace.append({'input_token':tid,'argmax_token':result,'uniform':uniform})
        return result
    def run_program(self,program,observer=None,reset=True):
        l=self.lay
        if not 1<=len(program)<=1000:raise ValueError('Expected 1..1000 gates')
        codes=[]
        for op in program:
            parts=op.split();g=(parts[0],*map(int,parts[1:]))
            if g not in l.gates:raise ValueError('Invalid gate/target/control')
            codes.append(l.gates.index(g))
        if not reset and self.cache is None:raise ValueError('No existing state to continue')
        if reset:self.cache=None
        for pc,code in enumerate(codes+[len(l.gates)]):self.step(l.load_start+pc*l.load_codes+code)
        self.step(l.reset_id if reset else l.resume_id)
        tid=l.controls[l.init_gate,0,1,0] if reset else l.read_start;trace=[]
        for _ in range((4*l.components+1)*(len(program)+2)+8):
            nxt=self.step(tid);trace.append({'input_token':tid,'argmax_token':nxt})
            if observer is not None and l.read_start<=tid<l.read_start+1024:
                observer(tid-l.read_start,self.diagnostic_state())
            if nxt==l.halt_id:self.trace.append(trace);return trace
            tid=nxt
        self.trace.append(trace);raise RuntimeError('Compact native controller did not HALT')
    def probability(self,event_mask,uniform=None):
        l=self.lay;mask=list(event_mask)
        if len(mask)!=l.size or any(type(x) not in (int,bool) or x not in (0,1) for x in mask):raise ValueError('Invalid mask')
        # A sample token reads the destination accumulator row. Its address
        # remains in the previous convolution tap. Flush that tap before the
        # reset's current-tap row-zero write, so the key stays one-hot.
        # HALT's embedding has only ballast; this is a native no-op forward.
        self.step(l.halt_id)
        self.step(l.prob_reset);trace=[];digits=[]
        for a,bit in enumerate(mask):
            digits=[]
            for ph in range(2):
                rid=4*a+2*ph+int(bit);start=l.query_select_start+rid
                query=self.step(start);trace.append([start,query])
                if not l.query_start+2*rid<=query<l.query_start+2*rid+2:raise RuntimeError('Invalid addressed query token')
                result=self.step(query);trace.append([query,result])
                if result//RADIX!=ph:raise RuntimeError('Invalid native probability digit')
                digits.append(result-ph*RADIX);self.step(result)
            if uniform is not None:
                outcome=self.step(l.sample_start+a,uniform);trace.append([l.sample_start+a,outcome])
                if l.outcome_start<=outcome<l.outcome_start+l.size:return dict(outcome=outcome-l.outcome_start,trace=trace)
                if outcome!=l.continue_id:raise RuntimeError('Invalid native sample decision')
        if uniform is not None:raise RuntimeError('CDF did not select an outcome')
        return dict(probability=(digits[0]*RADIX+digits[1])/Q-1,numeric_digits=digits,
                    numeric_token_ids=[digits[0],RADIX+digits[1]],trace=trace)
    def sample(self,uniforms):
        counts=collections.Counter()
        for u in uniforms:
            if not 0<float(u)<1:raise ValueError('Uniform must lie in (0,1)')
            a=self.probability([1]*self.lay.size,float(u))['outcome'];counts[format(a,f'0{self.lay.n}b')]+=1
        return dict(sorted(counts.items()))
    def diagnostic_state(self):
        """Scoring hook only; not read by the production execution path."""
        l=self.lay;state=self.cache.layers[0].recurrent_states[0]
        previous=float(state[0,2*l.controller_head,0,4])-C;source=1-int(round(previous))
        values=[]
        for p in range(l.components):
            bank,row=divmod(source*l.components+p,128)
            hi=float(state[0,2*bank,row,2])-C;lo=float(state[0,2*bank+1,row,2])-C
            values.append((hi*RADIX+lo)/Q-1)
        return torch.tensor(values).reshape(-1,2)
