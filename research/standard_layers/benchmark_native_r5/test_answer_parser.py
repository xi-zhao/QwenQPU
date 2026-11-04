import unittest
from answer_parser import parse_answer


class AnswerParserTests(unittest.TestCase):
    def test_event_inputs_are_not_answers(self):
        text='Event A: {"2":1}; Event B: {"0":1,"1":1}\n{"probabilities":{"A":0.5,"B":0.5}}'
        self.assertEqual(parse_answer(text)[0], {'probabilities': {'A': .5, 'B': .5}})
        with self.assertRaises(ValueError):
            parse_answer('{"A":{"2":1},"B":{"0":1,"1":1}}')

    def test_keep_conflicting_answer_failure(self):
        first='{"probabilities":{"A":0.5,"B":0.5}}'
        self.assertEqual(parse_answer(first+first)[0]['probabilities']['A'], .5)
        with self.assertRaises(ValueError):
            parse_answer(first+'{"probabilities":{"A":0,"B":0}}')

    def test_malformed_answer_is_not_silently_ignored(self):
        for text in ['{"probabilities":{"A":0.5}}', '{"probabilities":{"A":NaN,"B":0.5}}', '{"probabilities":{"A":true,"B":0.5}}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_answer(text)

    def test_supported_final_forms(self):
        self.assertEqual(parse_answer('{"A":0.5,"B":0.25}')[0]['probabilities']['B'], .25)
        self.assertEqual(parse_answer('A: 0.5\nB = 0.25')[1], 'labeled_decimal')
        with self.assertRaises(ValueError):
            parse_answer('A: 0.5\nA: 0\nB: 0.25')


if __name__ == '__main__':
    unittest.main()
