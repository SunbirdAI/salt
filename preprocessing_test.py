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

    def test_match_target_sentence_format_to_source(self):
        # Full_stop in source
        record = {'source': ['test sentence.'],
                  'target': ['translated sentence']}
        expected = ['translated sentence.']
        result = preprocessing.match_target_sentence_format_to_source(
            record, 'target')
        self.assertEqual(result['target'], expected)
        
        # Initial capital and no full stop
        record = {'source': ['Test sentence'],
                  'target': ['translated sentence.']}
        expected = ['Translated sentence']
        result = preprocessing.match_target_sentence_format_to_source(
            record, 'target')
        self.assertEqual(result['target'], expected)

    def test_clean_text(self):
        record = {'source': ['\\u2018Hello\\u2019 &lt;']}
        expected = ["'Hello' <"]
        result = preprocessing.clean_text(record, 'source')
        self.assertEqual(result['source'], expected)

    def test_random_case(self):
        record = {'source': ['Some words'], 'target': ['translated']}
        result = preprocessing.random_case(
            record, 'source', p_all_lower_case=0.0, p_all_upper_case=1.0,
            apply_to_both=True)
        self.assertEqual(result['source'], ['SOME WORDS'])
        self.assertEqual(result['target'], ['TRANSLATED'])
        
    def test_augment_characters(self):
        record = {'source': ['This is some sample source text']}
        result = preprocessing.augment_characters(
            record, 'source', avg_character_error_rate=0.5, p=1.0)
        # Check that augmentation occurred
        self.assertNotEqual(result['source'], record['source'])

    def test_augment_words(self):
        # Real usage of RandomWordAug
        record = {'source': ['source text']}
        word_augmentation_params = {'action': 'swap'}
        result = preprocessing.augment_words(
            record, 'source', **word_augmentation_params)
        self.assertNotEqual(['source text'], record['source'])
        self.assertEqual(len('source text'), len(record['source'][0]))

    def test_clean_and_remove_punctuation(self):
        record = {'source': ['&lt;hello,&gt; world!']}
        expected = ['hello world']
        result = preprocessing.clean_and_remove_punctuation(record, 'source')
        self.assertEqual(result['source'], expected)
        
        record = {'source': ["Hello! I'm here."]}
        expected = ["hello i'm here"]
        result = preprocessing.clean_and_remove_punctuation(
            record, 'source', allowed_punctuation="'")
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
    
    def test_audio_noise_on_empty_input(self):        
        # Test no error with empty input
        result = preprocessing.augment_audio_noise(
            {'source': [[]], 'source.sample_rate': [16000]}, 'source')
        
    def test_time_masking(self):
        result = preprocessing.augment_audio_time_masking(
            {'source': [np.ones(32000)],
             'source.sample_rate': [16000]}, 'source',
            num_masks_max=4, max_mask_duration_ms=100)
        self.assertTrue(np.sum(result['source'][0]) < 32000)
        self.assertTrue(np.sum(result['source'][0]) > 32000 - (16000 * .1 * 4))
        
        # Test no error with empty input
        result = preprocessing.augment_audio_time_masking(
            {'source': [[]], 'source.sample_rate': [16000]}, 'source')
        
    def test_speed_augmentation(self):
        # Test augmentation changes the length of data
        r = {'source': [np.ones(32000)], 'source.sample_rate': [16000]}
        result = preprocessing.augment_audio_speed(
            r, 'source', low=0.8, high=0.9)
        # Slow-down should result in longer audio array
        self.assertTrue(len(result['source'][0] > 32000))

        # Test no error with empty input
        r = {'source': [np.array([])], 'source.sample_rate': [16000]}
        result = preprocessing.augment_audio_speed(
            r, 'source', p=1.0, low=0.8, high=1.2)
        self.assertEqual(len(result['source'][0]), 0)
        
    def test_prefix_dataset_tag(self):
        record = {
            'source': ['test 1', 'test 2', 'test 3'],
            'source.origin_dataset': [
                'id-source1-train', 'id-source2-train', 'id-source1-train'
            ]}
        tags = {'source1': '<1>', 'source2': '<2>'}
        result = preprocessing.prefix_dataset_tag(
            record, 'source', **{'tags': tags})
        expected = ['<1> test 1', '<2> test 2', '<1> test 3']
        self.assertEqual(result['source'], expected)
               
if __name__ == '__main__':
    unittest.main()