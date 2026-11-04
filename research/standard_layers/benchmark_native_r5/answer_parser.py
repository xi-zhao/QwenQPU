"""Parse explicit probability answers without mistaking event masks for answers."""
import json
import math
import re


def _validate(values):
    if not isinstance(values, dict) or set(values) != {'A', 'B'}:
        raise ValueError('Expected probabilities A and B')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
        raise ValueError('Expected finite numerical probabilities')
    return {'probabilities': values}


def parse_answer(text):
    decoder = json.JSONDecoder()
    answers, pos = [], 0
    while pos < len(text):
        start = text.find('{', pos)
        if start < 0:
            break
        try:
            obj, length = decoder.raw_decode(text[start:])
        except ValueError:
            pos = start + 1
            continue
        pos = start + length
        if not isinstance(obj, dict):
            continue
        if 'probabilities' in obj:
            if set(obj) != {'probabilities'}:
                raise ValueError('Unexpected fields in probability answer')
            answers.append(_validate(obj['probabilities']))
        elif set(obj) == {'A', 'B'} and all(type(v) in (int, float) for v in obj.values()):
            answers.append(_validate(obj))
        # Event dictionaries have qubit-index keys, or dictionary-valued A/B.
        # They are inputs, not alternative numeric answers.
    if answers:
        if any(answer != answers[0] for answer in answers[1:]):
            raise ValueError('Conflicting probability answers')
        return answers[0], 'explicit_probability_json'
    found = re.findall(r'(?m)^\s*(?:[-*]\s*)?([AB])\s*[:=]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*[.;]?\s*$', text)
    values = {}
    for key, value in found:
        value = float(value)
        if key in values and values[key] != value:
            raise ValueError('Conflicting labeled answers')
        values[key] = value
    return _validate(values), 'labeled_decimal'
