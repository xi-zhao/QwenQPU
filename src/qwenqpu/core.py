"""1..16 qubits, <=1000 sequential unitary operations between RESETs.
Quantum runtime is a scripted recurrent nn.Module with local fixed 4x4 real gates.
State is [2**n,2] (real,imag), q0 least significant. No global dense gate matrices.
"""
import math
from typing import Tuple,List
import torch
from torch import nn,Tensor
import torch.nn.functional as F
class LocalGateStatevector(nn.Module):
    def __init__(self,n_qubits:int,dtype=torch.float32):
        super().__init__()
        if type(n_qubits) is not int or not 1<=n_qubits<=16:raise ValueError('n_qubits must be 1..16')
        self.n_qubits=n_qubits;self.dimension=1<<n_qubits;self.gate_count=0;self.terminal_readout=False
        h=1/math.sqrt(2);c=math.cos(math.pi/4);s=math.sin(math.pi/4)
        gates=[[[h,0,h,0],[0,h,0,h],[h,0,-h,0],[0,h,0,-h]],[[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]],[[1,0,0,0],[0,1,0,0],[0,0,c,-s],[0,0,s,c]]]
        self.gates=nn.Parameter(torch.tensor(gates,dtype=dtype),requires_grad=False)
        state=torch.zeros(self.dimension,2,dtype=dtype);state[0,0]=1
        self.register_buffer('state',state)
        self.register_buffer('pair_base',torch.arange(self.dimension//2,dtype=torch.long))
        self.register_buffer('bit_shifts',torch.arange(n_qubits-1,-1,-1,dtype=torch.long))
    def _valid(self,op:int,target:int,control:int)->bool:
        if op==0:return target==-1 and control==-1
        if not 1<=op<=4 or not 0<=target<self.n_qubits:return False
        if op==4:return 0<=control<self.n_qubits and control!=target
        return control==-1
    @torch.jit.export
    def reset_(self)->None:
        self.state.zero_();self.state[0,0]=1.;self.gate_count=0;self.terminal_readout=False
    @torch.jit.export
    def step(self,op:int,target:int,control:int=-1)->bool:
        if not self._valid(op,target,control):return False
        if op==0:self.reset_();return True
        if self.gate_count>=1000 or self.terminal_readout:return False
        low_mask=(1<<target)-1
        lo=(self.pair_base&low_mask)|((self.pair_base>>target)<<(target+1))
        hi=lo|(1<<target)
        addresses=torch.stack([lo,hi],dim=1).reshape(-1)
        pairs=self.state.index_select(0,addresses).reshape(-1,4)
        if op==4:
            updated=F.linear(pairs,self.gates[1])
            active=(((lo>>control)&1)!=0).unsqueeze(-1)
            updated=torch.where(active,updated,pairs)
        else:updated=F.linear(pairs,self.gates[op-1])
        self.state.index_copy_(0,addresses,updated.reshape(-1,2))
        self.gate_count+=1
        return True
    def forward(self,instructions:Tensor,reset:bool=False)->Tensor:
        return self.run(instructions,reset)
    @torch.jit.export
    def run(self,instructions:Tensor,reset:bool=False)->Tensor:
        # Compact instructions stay on CPU: integer graph control has no per-gate CUDA scalar synchronization.
        if instructions.dim()!=2 or instructions.size(1)!=3:raise ValueError('instructions shape must be [operations,3]')
        if instructions.dtype!=torch.int64 or instructions.device.type!='cpu':raise ValueError('instructions must be CPU int64')
        rows=instructions.size(0)
        if rows>1001:return torch.zeros(rows,dtype=torch.bool)
        count=0 if reset else self.gate_count;terminal=False if reset else self.terminal_readout
        for i in range(rows):
            op=int(instructions[i,0]);target=int(instructions[i,1]);control=int(instructions[i,2])
            if not self._valid(op,target,control):return torch.zeros(rows,dtype=torch.bool)
            if op==0:count=0;terminal=False
            else:
                if terminal or count>=1000:return torch.zeros(rows,dtype=torch.bool)
                count+=1
        if reset:self.reset_()
        for i in range(rows):
            if not self.step(int(instructions[i,0]),int(instructions[i,1]),int(instructions[i,2])):raise RuntimeError('preflight invariant')
        return torch.ones(rows,dtype=torch.bool)
    @torch.jit.export
    def raw_norm(self)->Tensor:
        return self.state.to(dtype=torch.float64).square().sum()
    @torch.jit.export
    def probabilities(self)->Tensor:
        raw=self.state.to(dtype=torch.float64).square().sum(dim=1)
        return raw/raw.sum().clamp_min(1e-300)
    @torch.jit.export
    def sample_terminal(self,uniforms:Tensor)->Tuple[Tensor,Tensor]:
        if uniforms.dim()!=1 or uniforms.numel()<1 or uniforms.numel()>10000:raise ValueError('supply 1..10000 terminal uniforms')
        valid=torch.isfinite(uniforms)&(uniforms>=0)&(uniforms<1)&torch.isfinite(self.raw_norm())&(self.raw_norm()>0)
        if not bool(valid.all()):return torch.full((uniforms.numel(),self.n_qubits),-1,dtype=torch.long,device=self.state.device),torch.zeros_like(valid)
        cdf=self.probabilities().cumsum(0);cdf=cdf/cdf[-1].clamp_min(1e-300)
        selected=torch.searchsorted(cdf,uniforms.to(dtype=torch.float64),right=True).clamp_max(self.dimension-1)
        bits=torch.bitwise_and(torch.bitwise_right_shift(selected.unsqueeze(1),self.bit_shifts),1)
        self.terminal_readout=True
        return bits,valid

def encode(operations,n_qubits):
    if type(n_qubits) is not int or not 1<=n_qubits<=16:raise ValueError('n_qubits 1..16')
    if type(operations) is not list or not 1<=len(operations)<=1001:raise ValueError('nonempty bounded instruction list required')
    rows=[];unitaries=0
    for text in operations:
        if type(text) is not str:raise ValueError('instructions must be strings')
        p=text.split(' ')
        if p==['RESET']:rows.append([0,-1,-1]);continue
        if len(p)==2 and p[0] in ['H','X','T'] and p[1].isdigit():
            t=int(p[1])
            if not 0<=t<n_qubits:raise ValueError('target out of range')
            rows.append([{'H':1,'X':2,'T':3}[p[0]],t,-1])
        elif len(p)==3 and p[0]=='CNOT' and p[1].isdigit() and p[2].isdigit():
            c,t=int(p[1]),int(p[2])
            if not(0<=c<n_qubits and 0<=t<n_qubits and c!=t):raise ValueError('invalid control/target')
            rows.append([4,t,c])
        else:raise ValueError('unsupported instruction '+repr(text))
        unitaries+=1
    if unitaries>1000:raise ValueError('limit is 1000 gate operations, not parallel depth')
    return torch.tensor(rows,dtype=torch.long)
