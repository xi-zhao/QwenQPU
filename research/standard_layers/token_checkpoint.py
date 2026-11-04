"""Small standard-layer QPU compiler and native numerical-token driver.

Only stock Qwen3_5ForCausalLM layers execute the register update, gate arithmetic,
probabilities and sample selection. The driver supplies instructions, chooses
argmax tokens, and supplies uniforms; it never computes amplitude updates.
This 1--4-qubit prototype is separate from the published 1--16 composite runtime.
"""
from dataclasses import dataclass
from pathlib import Path
import collections
import json
import math
import hashlib
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

B, C, REF, D, SCALE, Q = 2.0**40, 32., 2.0**24, 512, 2.0**-36, 2.0**22
RADIX = 4096
MASK = 2.0**28


@dataclass
class Layout:
    n: int
    autonomous: bool = False
    probability_readout: bool = False

    def __post_init__(self):
        if type(self.n) is not int or not 1 <= self.n <= 4:
            raise ValueError("This new-layer prototype accepts 1..4 qubits")
        self.size=1<<self.n
        self.components=2*self.size
        self.state_heads=2*self.components
        if self.probability_readout and not self.autonomous:
            raise ValueError('Probability readout requires the autonomous controller')
        self.heads=self.state_heads+(9 if self.autonomous else 0)+int(self.probability_readout)
        self.values=2*self.heads     # high and low digits of each component
        self.sel=6
        self.gate=self.sel+self.heads
        self.gates=[(g,q) for g in ["H","X","T"] for q in range(self.n)]
        self.gates += [("CNOT",c,t) for c in range(self.n) for t in range(self.n) if c!=t]
        self.ptr=self.gate+len(self.gates)
        self.src=self.ptr+self.components
        self.phase=self.src+2
        self.sample=self.phase+2
        self.uniform=self.sample+1
        self.read=self.uniform+1
        self.selected=self.read+self.values
        self.z=self.selected+self.components
        self.e=self.z+1
        self.prob=self.e+1
        self.score=self.prob+self.size
        self.penalty=self.score+self.size
        self.gray=self.penalty+3
        self.hidden=768 if self.autonomous else D
        self.ctrl=self.gray+12
        self.pc=self.ctrl+4
        self.rom=self.pc+1
        self.read_flag=self.rom+1
        self.load_flag=self.read_flag+1
        self.active_sample=self.load_flag+1
        self.ctrl_flag=self.active_sample+1
        self.rom_row=self.ctrl_flag+1
        self.rom_bank=self.rom_row+128
        self.next_pc=self.rom_bank+8
        self.last_low=self.next_pc+1
        self.halt_flag=self.last_low+1
        self.auto_penalty=self.halt_flag+1
        self.event_mask=self.auto_penalty+3
        self.probability_phase=self.event_mask+self.size
        self.probability_high=self.probability_phase+2
        self.probability_z=self.probability_high+1
        self.probability_gray=self.probability_z+1
        self.probability_penalty=self.probability_gray+12
        end=self.probability_penalty+2 if self.probability_readout else (self.auto_penalty+3 if self.autonomous else self.gray+12)
        assert end<self.hidden-1
        source=Path(__file__).resolve().parents[2]/'weight-release/module-bundle/quantum_gates.pt'
        raw=source.read_bytes()
        self.source_sha256=hashlib.sha256(raw).hexdigest()
        if self.source_sha256!='6aac7d79842243f8e58a698fa01936458e0e79079fd099d3cd6e05613145f34c':
            raise ValueError('Existing fixed-gate checkpoint identity changed')
        self.gate_weights=torch.load(source,weights_only=True,map_location='cpu')['gates']
        assert self.gate_weights.shape==(3,4,4) and self.gate_weights.dtype==torch.float32
        self.controls={}
        next_id=2*RADIX
        for g in range(len(self.gates)):
            for p in range(self.components):
                for source in range(2):
                    for phase in range(2):
                        self.controls[g,p,source,phase]=next_id;next_id+=1
        self.reset_control=next_id
        self.reset_high=next_id+1
        self.sample_ids=[next_id+2,next_id+3]
        self.outcome_start=next_id+4
        self.vocab=self.outcome_start+self.size
        if self.autonomous:
            self.read_start=self.vocab
            self.load_start=self.read_start+1024
            self.load_codes=len(self.gates)+1
            self.active_sample_id=self.load_start+1024*self.load_codes
            self.halt_id=self.active_sample_id+1
            self.vocab=self.halt_id+1
        if self.probability_readout:
            self.probability_query=[self.vocab,self.vocab+1]
            self.probability_result=self.vocab+2
            self.vocab=self.probability_result+2*RADIX

    def component(self, source, p):
        head=source*self.components+p
        return {self.read+2*head:RADIX/Q,
                self.read+2*head+1:1/Q,
                0:-1-C*(RADIX+1)/Q}

    def output_form(self, g, p, source):
        op=self.gates[g];a,ri=divmod(p,2)
        terms={}
        def add(component,coef):
            for d,w in self.component(source,component).items():terms[d]=terms.get(d,0.)+coef*w
        q=int(op[-1]);lo=a&~(1<<q);hi=lo|(1<<q)
        if op[0]=='CNOT':
            matrix=self.gate_weights[1] if a>>int(op[1])&1 else torch.eye(4)
        else:matrix=self.gate_weights[['H','X','T'].index(op[0])]
        r=2*((a>>q)&1)+ri
        for j in range(4):
            w=float(matrix[r,j])
            if w:add(2*(lo if j<2 else hi)+(j%2),w)
        return terms


def projection(dst,terms,multiplier=1.):
    for d,value in terms.items():dst[d]+=multiplier*value/(B*SCALE if d==0 else SCALE)


def compile_checkpoint(n=3,dtype=torch.float32,autonomous=False,stabilize_read_gates=True,probability_readout=False):
    """Construct weights only; no custom forward or kernel is attached."""
    lay=Layout(n,autonomous,probability_readout)
    need=2*len(lay.controls)+4*lay.components+12
    inter=((max(need,8190+4*lay.size+8)+127)//128)*128
    cfg=Qwen3_5TextConfig(vocab_size=lay.vocab,hidden_size=lay.hidden,intermediate_size=inter,
        num_hidden_layers=3+int(probability_readout),
        layer_types=["linear_attention"]+["full_attention"]*(2+int(probability_readout)),
        num_attention_heads=2,num_key_value_heads=1,head_dim=64,
        linear_num_key_heads=lay.heads,linear_num_value_heads=lay.values,
        linear_key_head_dim=128,linear_value_head_dim=128,tie_word_embeddings=False,
        attn_implementation="eager")
    cfg.qwenqpu_added_layer_prototype={"qubits":n,"format":"numeric-token-gray-v2",
        "digit_radix":RADIX,"amplitude_scale":Q,"language_weights_included":False,
        "source_quantum_gate_sha256":lay.source_sha256}
    cfg.qwenqpu_added_layer_prototype['autonomous_controller']=autonomous
    cfg.qwenqpu_added_layer_prototype['stabilized_memory_selector']=stabilize_read_gates
    cfg.qwenqpu_added_layer_prototype['probability_readout']=probability_readout
    model=Qwen3_5ForCausalLM(cfg).to(dtype).eval().requires_grad_(False)
    for p in model.parameters():p.zero_()
    emb=model.model.embed_tokens.weight
    emb[:,0:2]=B
    if autonomous:emb[:,-1]=B # 768 hidden / three ballast dimensions pins the same scale
    digits=torch.arange(RADIX,dtype=torch.float32)
    emb[:RADIX,2]=1;emb[:RADIX,4]=digits
    emb[RADIX:2*RADIX,3]=1;emb[RADIX:2*RADIX,4]=digits
    emb[lay.reset_control,lay.sel:lay.sel+lay.heads]=1
    emb[lay.reset_high,2]=1;emb[lay.reset_high,4]=1024;emb[lay.reset_high,5]=1
    for source,tid in enumerate(lay.sample_ids):emb[tid,lay.src+source]=1;emb[tid,lay.sample]=1
    for (g,p,source,phase),tid in lay.controls.items():
        emb[tid,lay.gate+g]=1;emb[tid,lay.ptr+p]=1
        emb[tid,lay.src+source]=1;emb[tid,lay.phase+phase]=1
        emb[tid,lay.sel+(1-source)*lay.components+p]=1
        if autonomous:emb[tid,lay.ctrl_flag]=1
    if autonomous:
        emb[lay.reset_control,lay.ctrl_flag]=1
        emb[lay.reset_control,lay.src+1]=1
        for pc in range(1024):
            tid=lay.read_start+pc
            emb[tid,lay.read_flag]=1;emb[tid,4]=pc
            emb[tid,lay.rom_row+pc%128]=1;emb[tid,lay.rom_bank+pc//128]=1
            for code in range(lay.load_codes):
                tid=lay.load_start+pc*lay.load_codes+code
                emb[tid,lay.load_flag]=1;emb[tid,4]=code
                emb[tid,lay.rom_row+pc%128]=1;emb[tid,lay.rom_bank+pc//128]=1
        emb[lay.active_sample_id,lay.sample]=1;emb[lay.active_sample_id,lay.active_sample]=1
    if probability_readout:
        for phase,tid in enumerate(lay.probability_query):
            emb[tid,lay.active_sample]=1
            emb[tid,lay.probability_phase+phase]=1
            rows=emb[lay.probability_result+phase*RADIX:lay.probability_result+(phase+1)*RADIX]
            rows[:,2+phase]=1;rows[:,4]=digits

    # GDN RAM: one address row per head; all arithmetic is in standard projections.
    la=model.model.layers[0].linear_attn
    kbase=lay.heads*128;vbase=2*kbase
    for head in range(lay.state_heads):
        projection(la.in_proj_qkv.weight[head*128],{0:128.})
        projection(la.in_proj_qkv.weight[kbase+head*128],{lay.sel+head:128.})
        la.conv1d.weight[head*128,0,-1]=1
        la.conv1d.weight[kbase+head*128,0,-2]=1 # previous instruction's address
        for digit in range(2):
            vh=2*head+digit;v=vbase+vh*128
            projection(la.in_proj_qkv.weight[v],{0:REF})
            projection(la.in_proj_qkv.weight[v+1],{0:REF})
            form={0:C,4:1.}
            if head%lay.components==0:form[5]=1024.
            projection(la.in_proj_qkv.weight[v+2],form)
            la.conv1d.weight[v:v+3,0,-1]=1
            projection(la.in_proj_b.weight[vh],{2+digit:256.,0:-128.})
            la.in_proj_z.weight[vh*128:vh*128+128,0]=128./(B*SCALE)
            la.out_proj.weight[lay.read+vh,vh*128+2]=REF/1024.
    la.dt_bias.fill_(-110.);la.norm.weight.fill_(1.)
    if autonomous:
        emit_program_memory(model,lay)
    if probability_readout:
        # Readout high digit only. Numeric feedback writes this register; the
        # event query has no amplitude address and cannot alter amplitudes.
        head=lay.heads-1
        projection(la.in_proj_qkv.weight[head*128],{0:128.})
        projection(la.in_proj_qkv.weight[kbase+head*128],{0:128.})
        la.conv1d.weight[head*128,0,-1]=1
        la.conv1d.weight[kbase+head*128,0,-1]=1
        vh=2*head;v=vbase+vh*128
        for j,form in enumerate([{0:REF},{0:REF},{0:C,4:1.}]):
            projection(la.in_proj_qkv.weight[v+j],form)
            la.conv1d.weight[v+j,0,-1]=1
        projection(la.in_proj_b.weight[vh],{2:256.,0:-128.})
        la.in_proj_z.weight[vh*128:vh*128+128,0]=128./(B*SCALE)
        la.out_proj.weight[lay.probability_high,vh*128+2]=REF/1024.

    m=model.model.layers[0].mlp;u=0
    def unit(gate,up,out,coef=1.):
        nonlocal u
        projection(m.gate_proj.weight[u],gate)
        projection(m.up_proj.weight[u],up)
        m.down_proj.weight[out,u]=coef;u+=1
    for (g,p,source,phase),tid in lay.controls.items():
        pred={lay.gate+g:512.,lay.ptr+p:512.,lay.src+source:512.,lay.phase+phase:512.,0:-1792.}
        e=lay.output_form(g,p,source)
        z={d:v*(Q/RADIX if phase==0 else Q) for d,v in e.items()}
        z[0]=z.get(0,0.)+(Q+.5)/RADIX-.5 if phase==0 else z.get(0,0.)+Q+C*RADIX
        if phase==1:z[lay.read+2*((1-source)*lay.components+p)]=-RADIX
        unit(pred,{d:v/256. for d,v in z.items()},lay.z)
        unit(pred,{d:v/256. for d,v in e.items()},lay.e)
    for source in range(2):
        for p in range(lay.components):
            unit({lay.src+source:128.},{d:v/128. for d,v in lay.component(source,p).items()},lay.selected+p)
    for j,flag in enumerate([lay.phase,lay.phase+1,lay.sample]):
        unit({0:128.},{0:-MASK/128.,flag:MASK/128.},lay.penalty+j)
    if autonomous:
        # Numeric feedback advances phase/pointer. ROM-read feedback starts the
        # next gate. These finite-state transitions are weight computations.
        form={lay.ctrl:4.*lay.components,lay.ctrl+1:4.,lay.ctrl+2:2.,lay.ctrl+3:3.,
              0:1-C*(4*lay.components+9)}
        unit({2:128.,3:128.},{d:v/128. for d,v in form.items()},lay.z)
        form={lay.rom:4.*lay.components,lay.ctrl+2:-2.,0:2+C*(2-4*lay.components)}
        unit({lay.read_flag:128.},{d:v/128. for d,v in form.items()},lay.z)
        unit({0:128.},{lay.pc:1/128.,0:(1-C)/128.},lay.next_pc)
        # After HALT, the current register is opposite the last gate's source.
        for source in range(2):
            condition={lay.active_sample:512.,lay.ctrl+2:512. if source==0 else -512.,
                       0:-768.-512*C if source==0 else -256.+512*C}
            for p in range(lay.components):
                if stabilize_read_gates:
                    # A plateau cancels small GDN read noise in a stored 0/1
                    # selector instead of multiplying that noise into amplitudes.
                    for shift,coef in [(128.,1.),(-128.,-1.)]:
                        gate=dict(condition);gate[0]+=shift
                        unit(gate,{d:v/256. for d,v in lay.component(source,p).items()},lay.selected+p,coef)
                else:
                    unit(condition,{d:v/256. for d,v in lay.component(source,p).items()},lay.selected+p)
    assert u<=inter

    # Born probabilities via SiLU(x)-SiLU(-x)=x, within the next standard MLP.
    m=model.model.layers[1].mlp;u=0
    for a in range(lay.size):
        for ri in range(2):
            d=lay.selected+2*a+ri
            for sign in [1.,-1.]:
                projection(m.gate_proj.weight[u],{d:sign})
                projection(m.up_proj.weight[u],{d:1.})
                m.down_proj.weight[lay.prob+a,u]=sign;u+=1
    # Gray-code interpolation avoids subtracting nearly equal million-sized
    # quadratic logits. Adjacent integer codes differ in one bit; its crossing
    # at z=k+1/2 selects the nearest digit through a small Hamming score.
    for bit in range(12):
        sequence=[((k^(k>>1))>>bit)&1 for k in range(RADIX)]
        slopes=[sequence[k+1]-sequence[k] for k in range(RADIX-1)]
        changes=[slopes[0]]+[slopes[k]-slopes[k-1] for k in range(1,RADIX-1)]+[-slopes[-1]]
        for knot,change in enumerate(changes):
            if not change:continue
            projection(m.gate_proj.weight[u],{lay.z:128.,0:-128.*knot})
            projection(m.up_proj.weight[u],{0:1/128.})
            m.down_proj.weight[lay.gray+bit,u]=change;u+=1
    if autonomous:
        # Last component/low digit switches from gate microcode to PC+1.
        last=lay.components-1;stride=lay.components
        form={lay.ctrl+1:1.,lay.ctrl+3:stride,2:2*stride,3:2*stride,
              0:-C*(1+stride)-last+.5-3*stride}
        for shift,coef in [(1.,1.),(0.,-1.)]:
            f={d:4*v for d,v in form.items()};f[0]=f.get(0,0.)+shift
            projection(m.gate_proj.weight[u],f,128.)
            projection(m.up_proj.weight[u],{0:1/128.})
            m.down_proj.weight[lay.last_low,u]=coef;u+=1
        # HALT is a ROM opcode, read through the same standard memory head.
        f={lay.rom:4.,lay.read_flag:512.,0:-4*(C+len(lay.gates))-512.}
        for shift,coef in [(2.,1.),(1.,-1.),(-1.,-1.),(-2.,1.)]:
            form=dict(f);form[0]+=shift
            projection(m.gate_proj.weight[u],form,128.)
            projection(m.up_proj.weight[u],{0:1/128.})
            m.down_proj.weight[lay.halt_flag,u]=coef;u+=1
    assert u<=inter
    # Interval scores select one sampled basis state through the normal lm_head.
    m=model.model.layers[2].mlp;u=0;sharp=2.0**24
    for a in range(lay.size):
        left={lay.prob+i:sharp for i in range(a)};left[lay.uniform]=-sharp
        right={lay.prob+i:-sharp for i in range(a+1)};right[lay.uniform]=sharp
        for form in [left,right]:
            projection(m.gate_proj.weight[u],form)
            projection(m.up_proj.weight[u],{0:1/sharp})
            m.down_proj.weight[lay.score+a,u]=-1.;u+=1
    if autonomous:
        forms=[{2:1.,3:1.,lay.last_low:-1.,lay.read_flag:1.,lay.halt_flag:-1.},
               {lay.last_low:1.},{lay.halt_flag:1.}]
        for j,valid in enumerate(forms):
            projection(m.gate_proj.weight[u],{0:128.})
            terms={d:MASK*v/128. for d,v in valid.items()};terms[0]=-MASK/128.
            projection(m.up_proj.weight[u],terms)
            m.down_proj.weight[lay.auto_penalty+j,u]=1.;u+=1
    if probability_readout:
        # Born probabilities from layer two are summed for a user-specified
        # 0/1 event mask. Phase-dependent scaling is performed by fixed weights.
        for phase,factor in [(0,Q/RADIX),(1,Q)]:
            for a in range(lay.size):
                projection(m.gate_proj.weight[u],{lay.event_mask+a:512.,lay.probability_phase+phase:512.,0:-768.})
                projection(m.up_proj.weight[u],{lay.prob+a:factor/256.})
                m.down_proj.weight[lay.probability_z,u]=1.;u+=1
        projection(m.gate_proj.weight[u],{lay.probability_phase:128.})
        projection(m.up_proj.weight[u],{0:(.5/RADIX-.5)/128.})
        m.down_proj.weight[lay.probability_z,u]=1.;u+=1
        projection(m.gate_proj.weight[u],{lay.probability_phase+1:128.})
        projection(m.up_proj.weight[u],{lay.probability_high:-RADIX/128.,0:C*RADIX/128.})
        m.down_proj.weight[lay.probability_z,u]=1.;u+=1
        assert u<=inter
        # A fourth ordinary decoder MLP converts the probability to numerical
        # tokens. No host read of amplitudes or probability tensors is needed.
        m=model.model.layers[3].mlp;u=0
        for bit in range(12):
            seq=[((k^(k>>1))>>bit)&1 for k in range(RADIX)]
            slopes=[seq[k+1]-seq[k] for k in range(RADIX-1)]
            changes=[slopes[0]]+[slopes[k]-slopes[k-1] for k in range(1,RADIX-1)]+[-slopes[-1]]
            for knot,change in enumerate(changes):
                if not change:continue
                projection(m.gate_proj.weight[u],{lay.probability_z:128.,0:-128.*knot})
                projection(m.up_proj.weight[u],{0:1/128.})
                m.down_proj.weight[lay.probability_gray+bit,u]=change;u+=1
        for phase in range(2):
            projection(m.gate_proj.weight[u],{0:128.})
            projection(m.up_proj.weight[u],{lay.probability_phase+phase:MASK/128.,0:-MASK/128.})
            m.down_proj.weight[lay.probability_penalty+phase,u]=1.;u+=1
        assert u<=inter

    lm=model.lm_head.weight
    for phase in range(2):
        rows=lm[phase*RADIX:(phase+1)*RADIX]
        gray=torch.arange(RADIX,dtype=torch.int64)
        gray=gray^(gray>>1)
        bits=torch.stack([(gray>>j)&1 for j in range(12)],dim=1)
        rows[:,lay.gray:lay.gray+12]=(2*bits-1).to(dtype)
        rows[:,0]=-bits.sum(1).to(dtype)/B
        rows[:,lay.penalty+phase]=1.
    for a in range(lay.size):
        lm[lay.outcome_start+a,lay.score+a]=2.0**20
        lm[lay.outcome_start+a,lay.penalty+2]=1.
    # Other vocabulary items are driver instructions, not model-generated results.
    lm[2*RADIX:lay.outcome_start,0]=-2*MASK/B
    if autonomous:
        # Control tokens use the same small Gray/Hamming readout as numbers.
        for code,tid in enumerate(range(2*RADIX,2*RADIX+len(lay.controls))):
            lm[tid].zero_();gray=code^(code>>1)
            lm[tid,lay.gray:lay.gray+12]=torch.tensor([1. if gray>>b&1 else -1. for b in range(12)])
            lm[tid,0]=-gray.bit_count()/B;lm[tid,lay.auto_penalty]=1.
        for pc in range(1024):
            tid=lay.read_start+pc;lm[tid,lay.next_pc]=2.*pc
            lm[tid,0]=-float(pc*pc)/B;lm[tid,lay.auto_penalty+1]=1.
        lm[lay.load_start:lay.active_sample_id,0]=-2*MASK/B
        lm[lay.active_sample_id,0]=-2*MASK/B
        lm[lay.halt_id,lay.auto_penalty+2]=1.
    if probability_readout:
        lm[lay.probability_query,0]=-2*MASK/B
        for phase in range(2):
            rows=lm[lay.probability_result+phase*RADIX:lay.probability_result+(phase+1)*RADIX]
            codes=torch.arange(RADIX,dtype=torch.int64);codes=codes^(codes>>1)
            bits=torch.stack([(codes>>j)&1 for j in range(12)],dim=1)
            rows[:,lay.probability_gray:lay.probability_gray+12]=(2*bits-1).to(dtype)
            rows[:,0]=-bits.sum(1).to(dtype)/B
            rows[:,lay.probability_penalty+phase]=1.
    return model,lay


def emit_program_memory(model,lay):
    """Store controller registers and a 1024-opcode ROM in DeltaNet states."""
    la=model.model.layers[0].linear_attn
    kbase=lay.heads*128;vbase=2*kbase
    def channel(rowid,terms,tap=-1):
        projection(la.in_proj_qkv.weight[rowid],terms)
        la.conv1d.weight[rowid,0,tap]=1.
    def value_head(vh,forms,beta):
        v=vbase+vh*128
        channel(v,{0:REF});channel(v+1,{0:REF})
        for j,form in enumerate(forms):channel(v+2+j,form)
        projection(la.in_proj_b.weight[vh],{beta:256.,0:-128.})
        la.in_proj_z.weight[vh*128:vh*128+128,0]=128./(B*SCALE)
    head=lay.state_heads
    channel(head*128,{0:128.})
    channel(kbase+head*128,{lay.ctrl_flag:128.,lay.read_flag:128.})
    forms=[{0:C,**{lay.gate+g:float(g) for g in range(len(lay.gates))}},
           {0:C,**{lay.ptr+p:float(p) for p in range(lay.components)}},
           {0:C,lay.src+1:1.},{0:C,lay.phase+1:1.}]
    value_head(2*head,forms,lay.ctrl_flag)
    value_head(2*head+1,[{0:C,4:1.}],lay.read_flag)
    for j in range(4):la.out_proj.weight[lay.ctrl+j,(2*head)*128+2+j]=REF/1024.
    la.out_proj.weight[lay.pc,(2*head+1)*128+2]=REF/1024.
    for bank in range(8):
        head=lay.state_heads+1+bank
        for rowid in range(128):
            form={lay.rom_row+rowid:128.,lay.rom_bank+bank:128.,0:-128.}
            channel(head*128+rowid,form)
            channel(kbase+head*128+rowid,form)
        value_head(2*head,[{0:C,4:1.}],lay.load_flag)
        la.out_proj.weight[lay.rom,(2*head)*128+2]=REF/1024.


class Driver:
    """Instruction scheduling and token feedback only; no state-vector arithmetic."""
    def __init__(self,model,lay):
        self.model,self.lay=model,lay
        self.cache=None;self.source=0;self.trace=[]
    def step(self,tid,uniform=None,event_mask=None):
        with torch.inference_mode():
            ids=torch.tensor([[tid]],device=next(self.model.parameters()).device)
            if uniform is None and event_mask is None:kwargs={"input_ids":ids}
            else:
                x=self.model.model.embed_tokens(ids).clone()
                if uniform is not None:x[0,0,self.lay.uniform]=uniform
                if event_mask is not None:
                    x[0,0,self.lay.event_mask:self.lay.event_mask+self.lay.size]=torch.tensor(event_mask,device=ids.device)
                kwargs={"inputs_embeds":x}
            out=self.model(**kwargs,past_key_values=self.cache,use_cache=True,logits_to_keep=1)
            self.cache=out.past_key_values
            return int(out.logits[0,-1].argmax())
    def reset(self):
        self.step(self.lay.reset_control);self.step(self.lay.reset_high)
        self.step(self.lay.reset_control);self.step(RADIX)
        self.source=0
    def gate(self,op):
        op=tuple(op.split()) if isinstance(op,str) else tuple(op)
        op=(op[0],*map(int,op[1:]))
        if op not in self.lay.gates:raise ValueError("Unsupported gate/target/control")
        g=self.lay.gates.index(op);generated=[]
        for p in range(self.lay.components):
            for phase in range(2):
                tid=self.lay.controls[g,p,self.source,phase]
                result=self.step(tid)
                if result//RADIX!=phase:raise RuntimeError("Native model produced an invalid numeric token")
                self.step(result);generated.append(result)
        self.source=1-self.source
        self.trace.append({"operation":list(op),"generated_numeric_token_ids":generated})
    def sample(self,uniforms):
        counts=collections.Counter()
        for value in uniforms:
            if not 0<float(value)<1:raise ValueError("Uniform must be in (0,1)")
            result=self.step(self.lay.sample_ids[self.source],float(value))-self.lay.outcome_start
            if not 0<=result<self.lay.size:raise RuntimeError("Native model produced invalid sample token")
            counts[format(result,f"0{self.lay.n}b")]+=1
        return dict(sorted(counts.items()))
    def diagnostic_state(self):
        """Read cache only for an independent test; never used by gate or sample."""
        state=self.cache.layers[0].recurrent_states[0]
        values=[]
        for p in range(self.lay.components):
            head=self.source*self.lay.components+p
            hi=float(state[0,2*head,0,2])-C
            lo=float(state[0,2*head+1,0,2])-C
            values.append((RADIX*hi+lo)/Q-1)
        return torch.tensor(values).reshape(-1,2)


def save(model,lay,path):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    model.save_pretrained(path,safe_serialization=True,max_shard_size="500MB")
    (path/"quantum_protocol.json").write_text(json.dumps({"qubits":lay.n,
        "format":model.config.qwenqpu_added_layer_prototype['format'],
        "program_loaded_by_host":True,"control_tokens_generated_internally":lay.autonomous,
        "numeric_writeback_and_sample_tokens_are_model_argmax":True},indent=2)+"\n")


class AutonomousDriver(Driver):
    """Load a ROM, then feed model-chosen tokens until its HALT token.

    No host PC increment, per-gate dispatch, amplitude computation or sampling
    decision occurs in this loop. Uniforms and final count aggregation are host work.
    """
    def __init__(self,model,lay):
        if not lay.autonomous:raise ValueError('Autonomous checkpoint required')
        super().__init__(model,lay)
        self.generated_trace=[]
    def run_program(self,program,observer=None):
        if not 1<=len(program)<=1000:raise ValueError('Program must have 1..1000 operations')
        codes=[]
        for op in program:
            parts=op.split();g=(parts[0],*map(int,parts[1:]))
            if g not in self.lay.gates:raise ValueError('Unsupported program instruction')
            codes.append(self.lay.gates.index(g))
        self.reset()
        for pc,code in enumerate(codes+[len(self.lay.gates)]):
            self.step(self.lay.load_start+pc*self.lay.load_codes+code)
        tid=self.lay.read_start
        trace=[]
        for _ in range((4*self.lay.components+1)*(len(program)+1)+8):
            nxt=self.step(tid)
            trace.append({'input_token':tid,'argmax_token':nxt})
            if observer is not None and self.lay.read_start<=tid<self.lay.read_start+1024:
                observer(tid-self.lay.read_start,self.diagnostic_state().clone())
            if nxt==self.lay.halt_id:
                self.generated_trace.append(trace)
                return trace
            tid=nxt
        self.generated_trace.append(trace)
        raise RuntimeError('Model did not reach HALT within the fixed step budget')
    def sample(self,uniforms):
        counts=collections.Counter()
        for value in uniforms:
            if not 0<float(value)<1:raise ValueError('Uniform must be in (0,1)')
            result=self.step(self.lay.active_sample_id,float(value))-self.lay.outcome_start
            if not 0<=result<self.lay.size:raise RuntimeError('Invalid model sample token')
            counts[format(result,f'0{self.lay.n}b')]+=1
        return dict(sorted(counts.items()))
    def probability(self,event_mask):
        """Return model-generated fixed-point digits as a normal numeric value.

        The host supplies the event definition and decodes its two output tokens.
        Born arithmetic, event sum and digit selection are checkpoint operations.
        """
        if not self.lay.probability_readout:raise ValueError('Checkpoint has no probability output')
        mask=list(event_mask)
        if len(mask)!=self.lay.size or any(type(x) not in (int,bool) or x not in (0,1) for x in mask):
            raise ValueError('Event mask must contain one 0/1 per basis state')
        high=self.step(self.lay.probability_query[0],event_mask=mask)
        high_digit=high-self.lay.probability_result
        if not 0<=high_digit<RADIX:raise RuntimeError('Invalid probability high-digit token')
        self.step(high)
        low=self.step(self.lay.probability_query[1],event_mask=mask)
        low_digit=low-self.lay.probability_result-RADIX
        if not 0<=low_digit<RADIX:raise RuntimeError('Invalid probability low-digit token')
        return {'probability':(high_digit*RADIX+low_digit)/Q,'numeric_token_ids':[high,low],
                'fixed_point_integer':high_digit*RADIX+low_digit,'fixed_point_scale':int(Q)}
    def diagnostic_state(self):
        state=self.cache.layers[0].recurrent_states[0]
        previous_source=float(state[0,2*self.lay.state_heads,0,4])-C
        self.source=1-int(round(previous_source))
        return super().diagnostic_state()
