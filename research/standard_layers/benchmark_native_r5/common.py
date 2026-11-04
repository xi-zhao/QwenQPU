"""Frozen output rules and budget accounting; no reference calculation."""
import json
import math
import re


class LimitReached(Exception):
    pass


def check_budget(language, native, limit):
    if language + native >= limit:
        raise LimitReached('prediction_budget')


def final_text(tokens, tokenizer, think_end=248069):
    if think_end not in tokens:
        raise ValueError('No end-of-thinking marker')
    boundary = len(tokens) - 1 - tokens[::-1].index(think_end)
    return tokenizer.decode(tokens[boundary + 1:], skip_special_tokens=True), boundary + 1


def object_from_text(text):
    decoder = json.JSONDecoder()
    objects, pos = [], 0
    while pos < len(text):
        start = text.find('{', pos)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except ValueError:
            pos = start + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        pos = start + end
    if not objects:
        raise ValueError('No JSON object')
    if any(x != objects[0] for x in objects[1:]):
        raise ValueError('Conflicting JSON objects')
    return objects[0]


def parse_answer(text):
    """Accept identical repeated JSON, or unambiguous labeled decimal answers."""
    try:
        obj = object_from_text(text)
    except ValueError as exc:
        if str(exc) != 'No JSON object':
            raise
        found = re.findall(r'(?m)^\s*(?:[-*]\s*)?([AB])\s*[:=]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*[.;]?\s*$', text)
        values = {}
        for name, value in found:
            value = float(value)
            if name in values and values[name] != value:
                raise ValueError('Conflicting labeled answers')
            values[name] = value
        obj = {'probabilities': values}
        mode = 'labeled_decimal'
    else:
        mode = 'json'
    if not isinstance(obj, dict) or set(obj) != {'probabilities'}:
        raise ValueError('Expected probabilities object')
    p = obj['probabilities']
    if not isinstance(p, dict) or set(p) != {'A', 'B'}:
        raise ValueError('Expected A and B')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in p.values()):
        raise ValueError('Nonfinite probability')
    return obj, mode
