"""Fixed output parsing; contains no quantum reference calculations."""
import json, math

def parse_json(text):
    decoder=json.JSONDecoder();objects=[];pos=0
    while pos<len(text):
        start=text.find('{',pos)
        if start<0:break
        try:obj,end=decoder.raw_decode(text[start:])
        except ValueError:pos=start+1;continue
        if isinstance(obj,dict):objects.append(obj)
        pos=start+end
    if not objects:raise ValueError('No JSON object')
    if any(x!=objects[0] for x in objects[1:]):raise ValueError('Conflicting JSON objects')
    return objects[0]

def probability_answer(obj):
    if not isinstance(obj,dict) or set(obj)!={'probabilities'}:raise ValueError('Expected probabilities object')
    p=obj['probabilities']
    if not isinstance(p,dict) or set(p)!={'A','B','C'}:raise ValueError('Expected named probabilities A, B, C')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in p.values()):raise ValueError('Probabilities must be finite numbers')
    return obj
