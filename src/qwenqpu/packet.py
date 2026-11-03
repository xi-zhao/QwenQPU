"""Host protocol parsing/scheduling only; no quantum arithmetic or outcome selection."""
from .core import encode

def parse_packet(packet):
 if type(packet) is not dict or set(packet)!={'n_qubits','prefix','circuit','repeat','shots'}:raise ValueError('Need exactly n_qubits,prefix,circuit,repeat,shots')
 n=packet['n_qubits'];repeat=packet['repeat'];shots=packet['shots'];prefix=packet['prefix'];pattern=packet['circuit']
 if type(n) is not int or not 1<=n<=16:raise ValueError('n_qubits must be integer 1..16')
 if type(repeat) is not int or not 1<=repeat<=1000:raise ValueError('repeat must be integer 1..1000')
 if type(shots) is not int or not 1<=shots<=10000:raise ValueError('shots must be integer 1..10000')
 if type(prefix) is not list or type(pattern) is not list or not pattern:raise ValueError('prefix list and nonempty circuit list required')
 if len(prefix)+len(pattern)*repeat>1000:raise ValueError('at most 1000 sequential gate operations')
 ops=prefix+pattern*repeat
 if any(type(op) is not str or op=='RESET' for op in ops):raise ValueError('unitary string gates only; initialization is a graph reset')
 return ops,encode(ops,n)
