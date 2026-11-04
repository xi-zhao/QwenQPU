import unittest
from tool_interface import parse_call

CALL='<tool_call><function=quantum_probabilities><parameter=circuit>["H 0"]</parameter><parameter=events>{"A":{"0":1},"B":{"0":0}}</parameter></function></tool_call>'


class InterfaceTests(unittest.TestCase):
    def test_wait_for_real_handoff(self):
        self.assertIsNone(parse_call(CALL))
        self.assertIsNone(parse_call('</think>'+CALL[:-12]))
        self.assertEqual(parse_call('Reasoning ends.</think>'+CALL)['circuit'],['H 0'])

    def test_do_not_execute_analysis_example(self):
        self.assertIsNone(parse_call(CALL+'</think>Still no call'))

    def test_non_thinking_template_handoff(self):
        self.assertEqual(parse_call(CALL, thinking_ended_in_prompt=True)['circuit'], ['H 0'])
        self.assertIsNone(parse_call('<think>'+CALL, thinking_ended_in_prompt=True))
        self.assertEqual(parse_call('<think>analysis</think>'+CALL, thinking_ended_in_prompt=True)['circuit'], ['H 0'])

    def test_strict_parameters(self):
        with self.assertRaises(ValueError):parse_call('</think>'+CALL.replace('quantum_probabilities','unrelated_tool'))
        with self.assertRaises(ValueError):parse_call('</think>'+CALL.replace('<parameter=events>','<parameter=circuit>'))
        with self.assertRaises(ValueError):parse_call('</think>'+CALL.replace('["H 0"]','__import__("os")'))


if __name__=='__main__':unittest.main()
