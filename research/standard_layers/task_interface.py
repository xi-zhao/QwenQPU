"""Task execution and numeric output, without a model self-evaluation round.

The host parses requested events and decodes numeric tokens. The checkpoint
executes the circuit, Born arithmetic and event sums. Reference answers and
correctness labels belong to a separate evaluator.
"""


def event_mask(pattern,n):
    if isinstance(pattern,dict):
        # A qubit-indexed conjunction avoids asking the language model to
        # reverse a bit string. Omitted qubits are marginalized over.
        if any(not isinstance(k,str) or k not in {str(q) for q in range(n)}
               or type(v) is not int or v not in (0,1) for k,v in pattern.items()):
            raise ValueError('Event keys must be qubit indices and values must be 0 or 1')
        return [int(all((i >> int(q)) & 1 == v for q,v in pattern.items())) for i in range(1<<n)]
    if not isinstance(pattern,str) or len(pattern)!=n or any(c not in '01?' for c in pattern):
        raise ValueError('Event pattern must have n characters from 0, 1, ?; rightmost is q0')
    return [int(all(a=='?' or a==b for a,b in zip(pattern,format(i,f'0{n}b')))) for i in range(1<<n)]


def execute_probability_task(driver,request):
    """Return requested probabilities and a separate execution trace.

    request = {"circuit":["H 0","T 0","H 0"],
               "events":{"P000":"000","q0=1":"??1"}}
    """
    if not isinstance(request,dict) or set(request)!={'circuit','events'}:
        raise ValueError('Expected exactly circuit and events')
    circuit,events=request['circuit'],request['events']
    if not isinstance(circuit,list) or not all(isinstance(op,str) for op in circuit):
        raise ValueError('Circuit must be an instruction list')
    if not isinstance(events,dict) or not 1<=len(events)<=1<<driver.lay.n:
        raise ValueError('Expected 1..2**n named events')
    if not all(isinstance(k,str) and k for k in events):
        raise ValueError('Event names must be nonempty strings')
    masks={name:event_mask(pattern,driver.lay.n) for name,pattern in events.items()}
    trace=driver.run_program(circuit)
    raw={name:driver.probability(mask) for name,mask in masks.items()}
    answer={'probabilities':{name:result['probability'] for name,result in raw.items()}}
    evidence={'native_trace':trace,'numeric_readout':raw,'request':request}
    return answer,evidence
