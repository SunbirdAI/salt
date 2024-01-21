import unittest
import re
import numpy as np

from . import preprocessing

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.record = {
            'source': ['test sentence'],
            'source.language': ['eng'],
            'source.origin_dataset': ['salt'],
            'target': [[0.1, 0.2, 0.3]],
            'target.sample_rate': [16000],
            'target.language': ['lug'],
            'target.is_studio': [False],
        }

    def test_prefix_target_language(self):
        expected = ['>>lug<< test sentence']
        result = preprocessing.prefix_target_language(self.record, 'source')
        self.assertEqual(result['source'], expected)

    def test_sentence_format(self):
        # Test with default add_full_stop
        record = {'source': ['test sentence']}
        expected = ['Test sentence.']
        result = preprocessing.sentence_format(record, 'source')
        self.assertEqual(result['source'], expected)

        # Test without full stop
        record = {'source': ['test sentence']}
        expected = ['Test sentence']
        result = preprocessing.sentence_format(
            record, 'source', add_full_stop=False)
        self.assertEqual(result['source'], expected)

    def test_normalise_text(self):
        record = {'source': ['“Hello, World!”']}
        expected = ['"Hello, World!"']
        result = preprocessing.normalise_text(record, 'source')
        self.assertEqual(result['source'], expected)

    def test_augment_characters(self):
        record = {'source': ['source text']}
        char_augmentation_params = {'action': 'swap'}
        result = preprocessing.augment_characters(
            record, 'source', **char_augmentation_params)
        # Check that augmentation occurred
        self.assertNotEqual(['source text'], record['source'])
        # Character swap, so length should be unchanged
        self.assertEqual(len('source text'), len(record['source'][0]))

    def test_augment_words(self):
        # Real usage of RandomWordAug
        record = {'source': ['source text']}
        word_augmentation_params = {'action': 'swap'}
        result = preprocessing.augment_words(
            record, 'source', **word_augmentation_params)
        self.assertNotEqual(['source text'], record['source'])
        self.assertEqual(len('source text'), len(record['source'][0]))

    def test_remove_punctuation(self):
        record = {'source': ['hello, world!']}
        expected = ['hello world']
        result = preprocessing.remove_punctuation(record, 'source')
        self.assertEqual(result['source'], expected)

    def test_lower_case(self):
        record = {'source': ['HELLO, WoRld.']}
        expected = ['hello, world.']
        result = preprocessing.lower_case(record, 'source')
        self.assertEqual(result['source'], expected)

    def test_lower_case_batch(self):
        record = {'source': ['HELLO WoRld', 'More TEXT', 'AnotheR']}
        expected = ['hello world', 'more text', 'another']
        result = preprocessing.lower_case(record, 'source')
        self.assertEqual(result['source'], expected)
        
    def test_resample_audio(self):
        result = preprocessing.set_sample_rate(
            self.record, 'target', rate=32_000)
        self.assertEqual(len(result['target'][0]), 6)
        
         
if __name__ == '__main__':
    unittest.main()