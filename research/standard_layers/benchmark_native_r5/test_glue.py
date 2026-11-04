import unittest
from common import LimitReached, check_budget, object_from_text
from answer_parser import parse_answer
from tool_interface import response_text
from score import cp_bounds


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        values = {1:'{"probabilities":{"A":0,"B":0}}', 2:'{"probabilities":{"A":0.5,"B":1}}'}
        if not skip_special_tokens:
            values.update({248068:'<think>',248069:'</think>'})
        return ''.join(values.get(i,'') for i in ids)


class GlueTests(unittest.TestCase):
    def test_only_final_after_thinking(self):
        text, count = response_text([248068,1,248069,2,248046],FakeTokenizer())
        self.assertEqual(parse_answer(text)[0]['probabilities'], {'A':.5,'B':1})
        self.assertEqual(count,3)
        with self.assertRaises(ValueError):response_text([248068,1,248046],FakeTokenizer())
        self.assertEqual(response_text([2,248046],FakeTokenizer())[1],0)

    def test_no_reference_dependent_repair(self):
        value='{"probabilities":{"A":0.5,"B":1}}'
        self.assertEqual(parse_answer(value+value)[1],'explicit_probability_json')
        self.assertEqual(parse_answer('A: 0.5\nB = 1.0')[1],'labeled_decimal')
        with self.assertRaises(ValueError):parse_answer(value+'{"probabilities":{"A":0,"B":0}}')
        with self.assertRaises(ValueError):parse_answer('A: 0.5\nA: 0\nB: 1')

    def test_cumulative_budget(self):
        check_budget(384,15999,16384)
        with self.assertRaises(LimitReached):check_budget(384,16000,16384)
        with self.assertRaises(LimitReached):check_budget(16384,0,16384)
        with self.assertRaises(LimitReached):check_budget(0,16384,16384)

    def test_fake_pipeline(self):
        request=object_from_text('{"circuit":["H 0"],"events":{"A":{"0":1},"B":{"0":0}}}')
        calls=[]
        for i in range(4):
            check_budget(3,len(calls),7);calls.append(i)
        self.assertEqual(request['events']['A'],{'0':1})
        with self.assertRaises(LimitReached):check_budget(3,len(calls),7)
        self.assertEqual(parse_answer('{"probabilities":{"A":0.5,"B":0.5}}')[0]['probabilities']['A'],.5)

    def test_non_degenerate_exact_intervals(self):
        self.assertEqual(cp_bounds(0,12)[0],0)
        self.assertGreater(cp_bounds(0,12)[1],0)
        self.assertLess(cp_bounds(12,12)[0],1)
        self.assertEqual(cp_bounds(12,12)[1],1)


if __name__=='__main__':unittest.main()
