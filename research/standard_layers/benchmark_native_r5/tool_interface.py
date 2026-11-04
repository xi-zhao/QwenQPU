"""Qwen3.5's observed native function/parameter tool-call format."""
import json
import re

SYSTEM = (
    'Solve the supplied quantum-circuit probability task. '
    'The final answer is {"probabilities":{"A":number,"B":number}}. '
    'When quantum_probabilities is available, immediately call it with the supplied circuit and event conditions unchanged. '
    'The function calculates the probabilities; submit its arguments before doing any probability calculation yourself. '
    'Its numerical response completes the task. Without that function, calculate the probabilities yourself. '
    'Give finite decimal probabilities in the final answer. '
    'Use at least six decimal places when a probability is not exactly representable by a shorter decimal.'
)

TOOL = {'type':'function','function':{
    'name':'quantum_probabilities',
    'description':'Execute a supplied H/X/T/CNOT circuit from the all-zero state and return the requested final measurement probabilities. The function response is the final numerical answer.',
    'parameters':{'type':'object','properties':{
        'circuit':{'type':'array','items':{'type':'string'},'description':'JSON array of gate strings in the supplied order, such as ["H 0","CNOT 0 1"]. CNOT operands are control then target.'},
        'events':{'type':'object','properties':{name:{'type':'object','properties':{str(q):{'type':'integer','enum':[0,1]} for q in range(6)},'additionalProperties':False} for name in ['A','B']},
                  'required':['A','B'],'additionalProperties':False,
                  'description':'JSON object mapping A and B to qubit conditions, for example {"A":{"2":1},"B":{"0":0,"1":1}}. Entries are measurement bit values; unspecified qubits are marginalized.'}},
        'required':['circuit','events'],'additionalProperties':False}}}


def parse_call(text, thinking_ended_in_prompt=False):
    """Return a completed call only after the thinking block closes."""
    if '</think>' in text:
        final=text.rsplit('</think>',1)[1]
    elif thinking_ended_in_prompt and '<think>' not in text:
        final=text
    else:
        return None
    if '<tool_call>' not in final or '</tool_call>' not in final:
        return None
    start=final.index('<tool_call>')+len('<tool_call>')
    end=final.index('</tool_call>',start)
    body=final[start:end].strip()
    function=re.fullmatch(r'<function=quantum_probabilities>\s*(.*?)\s*</function>',body,re.S)
    if not function:raise ValueError('Expected native quantum_probabilities function block')
    parameters={};pos=0
    for match in re.finditer(r'<parameter=([A-Za-z_]+)>\s*(.*?)\s*</parameter>',function[1],re.S):
        if function[1][pos:match.start()].strip():raise ValueError('Unexpected text in function block')
        name,value=match[1],match[2]
        if name in parameters:raise ValueError('Repeated function parameter')
        parameters[name]=json.loads(value);pos=match.end()
    if function[1][pos:].strip() or set(parameters)!={'circuit','events'}:
        raise ValueError('Expected exactly circuit and events parameters')
    return parameters


def prompt(task):
    return (f'A {task["qubits"]}-qubit register starts in the all-zero state. '
            'Apply this circuit in the listed order: '+json.dumps(task['circuit'])+'. '
            'Find the probabilities of these computational-basis measurement events: '+json.dumps(task['events'])+'. '
            'Each event is a conjunction of indexed bit values; unspecified qubits are marginalized. CNOT is control then target.')


def response_text(tokens, tokenizer):
    """The non-thinking template closes the empty thinking block in the prompt."""
    text=tokenizer.decode(tokens,skip_special_tokens=False)
    if '</think>' in text:
        boundary=len(tokens)-1-tokens[::-1].index(248069)
        return tokenizer.decode(tokens[boundary+1:],skip_special_tokens=True),boundary+1
    if '<think>' in text:
        raise ValueError('Model reopened thinking without closing it')
    return tokenizer.decode(tokens,skip_special_tokens=True),0
