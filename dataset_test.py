import unittest
import unittest.mock
import soundfile
import pandas as pd
import datasets
import tempfile
import os
import numpy as np
import yaml
import random
import string
import warnings

from . import dataset


@unittest.mock.patch.dict(os.environ, {"HF_DATASETS_DISABLE_PROGRESS_BARS": "1"})
class DatasetTestCase(unittest.TestCase):
    def assertNestedAlmostEqual(self, expected, actual, places=3):
        '''
        Recursive function to compare nested data structures with support for
        comparing floating point numbers using assertAlmostEqual and ignores
        differences in base temporary path of file paths.
        '''
        if isinstance(expected, (float, np.floating)):
            self.assertAlmostEqual(expected, actual, places=places)
        elif isinstance(expected, list):
            self.assertEqual(len(expected), len(actual))
            for exp, act in zip(expected, actual):
                self.assertNestedAlmostEqual(exp, act, places=places)
        elif isinstance(expected, dict):
            self.assertEqual(set(expected.keys()), set(actual.keys()))
            for key in expected:
                if key == 'path':
                  # Paths don't need to match
                  pass
                else:
                    self.assertNestedAlmostEqual(expected[key], actual[key],
                                                 places=places)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_almost_equal(expected, actual, decimal=places)
        else:
            self.assertEqual(expected, actual)
            
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = self.temp_dir.name

        audio_path = f'{self.data_path}/audio'
        if not os.path.exists(audio_path):
            os.mkdir(audio_path)
            
        soundfile.write(
          f'{audio_path}/lug1.wav', np.array([.1, .1, .1], np.float32), 16000)
        soundfile.write(
          f'{audio_path}/lug2.wav', np.array([.2, .2, .2], np.float32), 16000)
        soundfile.write(
          f'{audio_path}/lug1_studio.wav', np.array([.3, .3, .3], np.float32),
          16000)
        soundfile.write(
          f'{audio_path}/lug2_studio.wav', np.array([.4, .4, .4], np.float32),
          16000)
        soundfile.write(
          f'{audio_path}/eng1.wav', np.array([.5, .5, .5], np.float32), 16000)
        soundfile.write(
          f'{audio_path}/eng2.wav', np.array([.6, .6, .6], np.float32), 16000)
        soundfile.write(
          f'{audio_path}/eng3.wav', np.array([.7, .7, .7], np.float32), 16000)
        soundfile.write(
          f'{audio_path}/ach1.wav', np.array([.8, .8, .8], np.float32), 16000)
        
        translate_data_1 = {
            'id': [1, 2, 3],
            'lug_text': ['lug1', 'lug2', 'lug3'],
            'ach_text': ['ach1', 'ach2', 'ach3'],
            'eng_text': ['eng1', 'eng2', 'eng3'],
        }

        translate_data_2 = {
            'id': [4, 5],
            'lug_text': ['lug4', 'lug5'],
            'eng_text': ['eng4', 'eng5'],
        }
        
        translate_data_missing_value = {
            'id': [1, 2, 3],
            'lug_text': ['lug1', None, 'lug3'],
            'ach_text': ['ach1', 'ach2', 'ach3'],
            'eng_text': ['eng1', 'eng2', 'eng3'],
        }
        
        audio_metadata = {
            'id': [1, 1, 1, 1, 2, 2, 2, 3,],
            'audio': [
                f'{audio_path}/lug1.wav',
                f'{audio_path}/lug1_studio.wav',
                f'{audio_path}/eng1.wav',
                f'{audio_path}/ach1.wav',
                f'{audio_path}/lug2.wav',
                f'{audio_path}/lug2_studio.wav',
                f'{audio_path}/eng2.wav',
                f'{audio_path}/eng3.wav',
                
            ],
            'text': [
              'lug1', 'lug1', 'eng1', 'ach1', 'lug2',  'lug2',  'eng2', 'eng3'],
            'audio_language': [
              'lug', 'lug', 'eng', 'ach', 'lug', 'lug', 'eng', 'ach'],
            'is_studio': [
              False, False, True, True, False, False, False, False],
            'speaker_id': [
                'lug-001',
                'lug-studio-1',
                'eng-001',
                'ach-001',
                'lug-002',
                'lug-studio-1',
                'eng-002',
                'eng-001',
            ]   
        }

        audio_metadata_unsorted = {
            'id': [1, 2, 1, 2, 1, 2, 3, 1],
            'audio': [
                f'{audio_path}/lug1.wav',
                f'{audio_path}/lug2.wav',
                f'{audio_path}/lug1_studio.wav',
                f'{audio_path}/lug2_studio.wav',
                f'{audio_path}/eng1.wav',
                f'{audio_path}/eng2.wav',
                f'{audio_path}/eng3.wav',
                f'{audio_path}/ach1.wav',
            ],
            'text': [
              'lug1', 'lug2', 'lug1', 'lug2', 'eng1', 'eng2', 'eng3', 'ach1'],
            'audio_language': [
              'lug', 'lug', 'lug', 'lug', 'eng', 'eng', 'eng', 'ach'],
            'is_studio': [
              False, False, True, True, False, False, False, False],
            'speaker_id': [
                'lug-001',
                'lug-002',
                'lug-studio-1',
                'lug-studio-1',
                'eng-001',
                'eng-002',
                'eng-001',
                'ach-001',
            ]
        }

        temp_csv_path = f'{self.data_path}/translation_dataset_1.csv'
        pd.DataFrame(translate_data_1).to_csv(temp_csv_path, index=False)    

        temp_csv_path = f'{self.data_path}/translation_dataset_2.csv'
        pd.DataFrame(translate_data_2).to_csv(temp_csv_path, index=False)
        
        temp_csv_path = f'{self.data_path}/translation_missing_value.csv'
        pd.DataFrame(translate_data_missing_value).to_csv(
          temp_csv_path, index=False)

        audio_dataset = datasets.Dataset.from_dict(audio_metadata).cast_column(
          'audio', datasets.Audio(sampling_rate=16000))
        audio_dataset.to_parquet(f'{self.data_path}/audio_mock.parquet')

        audio_dataset_unsorted = datasets.Dataset.from_dict(audio_metadata_unsorted).cast_column(
            'audio', datasets.Audio(sampling_rate=16000))
        audio_dataset_unsorted.to_parquet(
            f'{self.data_path}/audio_mock_unsorted.parquet')
        
        datasets.disable_progress_bar()
        # HuggingFace datasets gives a ResourceWarning with temp files
        warnings.simplefilter("ignore", ResourceWarning)
        
    def tearDown(self):
        self.temp_dir.cleanup()
        warnings.simplefilter("default", ResourceWarning)
      

    #TODO: audio files of different sample rates
    def test_preprocessing_augmentation(self):
        def random_prefix(r, src_or_tgt):
            for i in range(len(r['source'])):
                prefix = ''.join(
                    random.choice(string.ascii_letters) for _ in range(6))
                r[src_or_tgt][i] = f'>{prefix}< ' + r[src_or_tgt][i]
            return r
        
        setattr(dataset.preprocessing, 'random_prefix', random_prefix)
        
        yaml_config = '''
        huggingface_load:
            path: csv
            data_files: PATH/translation_dataset_1.csv
            split: train
        source:
            type: text
            language: lug
            preprocessing:
                - random_prefix
        target:
            type: text
            language: eng
        '''.replace('PATH', self.data_path)

        config = yaml.safe_load(yaml_config)
        ds = dataset.create(config)
        
        sample_1 = list(ds)
        sample_2 = list(ds)
        
        # Check we get a different random example each time the dataset is read
        self.assertNotEqual(sample_1[0]['source'], sample_2[0]['source'])
        self.assertEqual(sample_1[0]['target'], sample_2[0]['target'])
        
    def test_preprocessing_pipeline(self):
        def prefix(r, src_or_tgt, tag):
            for i in range(len(r['source'])):
                r[src_or_tgt][i] = tag + ' ' + r[src_or_tgt][i]
            return r

        def suffix(r, src_or_tgt, tag):
            for i in range(len(r['source'])):
                r[src_or_tgt][i] = r[src_or_tgt][i] + ' ' + tag
            return r
        
        setattr(dataset.preprocessing, 'prefix', prefix)
        setattr(dataset.preprocessing, 'suffix', suffix)
        
        yaml_config = '''
        huggingface_load:
            path: csv
            data_files: PATH/translation_dataset_1.csv
            split: train
        source:
            type: text
            language: lug
            preprocessing:
                - prefix:
                    tag: one
                - prefix:
                    tag: two
                - suffix:
                    tag: three
        target:
            type: text
            language: eng
            preprocessing:
                - suffix:
                    tag: four
        '''.replace('PATH', self.data_path)

        config = yaml.safe_load(yaml_config)
        ds = dataset.create(config)
        
        expected = [
            {'source': 'two one lug1 three', 'target': 'eng1 four'},
            {'source': 'two one lug2 three', 'target': 'eng2 four'},
            {'source': 'two one lug3 three', 'target': 'eng3 four'}]

        self.assertEqual(list(ds), expected)
   
    def test_text_to_speech_dataset(self):      
      yaml_config = '''
      huggingface_load:
          join:
            - path: parquet
              data_files: PATH/audio_mock.parquet
              split: train
            - path: csv
              data_files: PATH/translation_dataset_1.csv
              split: train
      source:
          type: text
          language: lug
      target:
          type: speech
          language: lug
          speaker_id: lug-studio-1
      '''.replace('PATH', self.data_path)
      config = yaml.safe_load(yaml_config)
      ds = dataset.create(config)
      
      expected = [
        {'source': 'lug1',
         'target': {'path': None,
                    'array': np.array([.3, .3, .3]),
                    'sampling_rate': 16000}},
        {'source': 'lug2',
         'target': {'path': None,
                    'array': np.array([.4, .4, .4]),
                    'sampling_rate': 16000}},
      ]
      
      self.assertNestedAlmostEqual(list(ds), expected)

        
    def test_single_dataset(self):        
        yaml_config = '''
        huggingface_load:
            path: csv
            data_files: PATH/translation_dataset_1.csv
            split: train
        source:
            type: text
            language: lug
        target:
            type: text
            language: eng
        '''.replace('PATH', self.data_path)

        config = yaml.safe_load(yaml_config)
        ds = dataset.create(config)
        
        expected = [
            {'source': 'lug1', 'target': 'eng1'},
            {'source': 'lug2', 'target': 'eng2'},
            {'source': 'lug3', 'target': 'eng3'}]

        self.assertEqual(list(ds), expected)
   
    def test_translation_multiple_to_one(self):        
        yaml_config = '''
        huggingface_load:
            path: csv
            data_files: PATH/translation_dataset_1.csv
            split: train
        source:
            type: text
            language: [lug, ach]
        target:
            type: text
            language: eng
        '''.replace('PATH', self.data_path)

        config = yaml.safe_load(yaml_config)
        ds = dataset.create(config)
        
        expected = [
            {'source': 'lug1', 'target': 'eng1'},
            {'source': 'lug2', 'target': 'eng2'},
            {'source': 'lug3', 'target': 'eng3'},
            {'source': 'ach1', 'target': 'eng1'},
            {'source': 'ach2', 'target': 'eng2'},
            {'source': 'ach3', 'target': 'eng3'},
        ]

        self.assertCountEqual(list(ds), expected)

    def test_two_datasets_concatenated(self):        
        yaml_config = '''
        huggingface_load:
          - path: csv
            data_files: PATH/translation_dataset_1.csv
            split: train
          - path: csv
            data_files: PATH/translation_dataset_2.csv
            split: train
        source:
            type: text
            language: lug
        target:
            type: text
            language: eng
        '''.replace('PATH', self.data_path)

        config = yaml.safe_load(yaml_config)
        ds = dataset.create(config)
        
        expected = [
            {'source': 'lug1', 'target': 'eng1'},
            {'source': 'lug2', 'target': 'eng2'},
            {'source': 'lug3', 'target': 'eng3'},
            {'source': 'lug4', 'target': 'eng4'},
            {'source': 'lug5', 'target': 'eng5'}
        ]

        self.assertEqual(list(ds), expected)

    def test_missing_value(self):        
        yaml_config = '''
        huggingface_load:
          - path: csv
            data_files: PATH/translation_missing_value.csv
            split: train
          - path: csv
            data_files: PATH/translation_dataset_2.csv
            split: train
        source:
            type: text
            language: lug
        target:
            type: text
            language: eng
        '''.replace('PATH', self.data_path)

        config = yaml.safe_load(yaml_config)
        ds = dataset.create(config)
        
        expected = [
            {'source': 'lug1', 'target': 'eng1'},
            {'source': 'lug3', 'target': 'eng3'},
            {'source': 'lug4', 'target': 'eng4',},
            {'source': 'lug5', 'target': 'eng5'}
        ]

        self.assertEqual(list(ds), expected)
            
    def test_speech_dataset(self):
      yaml_config = '''
      huggingface_load:
          path: parquet
          data_files: PATH/audio_mock_unsorted.parquet
          split: train
      source:
          type: speech
          language: lug
      target:
          type: text
          language: lug
      '''.replace('PATH', self.data_path)
      config = yaml.safe_load(yaml_config)
      ds = dataset.create(config)
      
      expected = [
        {'source': {'path': None,
                    'array': np.array([.1, .1, .1]),
                    'sampling_rate': 16000},
         'target': 'lug1'},
        {'source': {'path': None,
                    'array': np.array([.2, .2, .2]),
                    'sampling_rate': 16000},
         'target': 'lug2'},
        {'source': {'path': None,
                    'array': np.array([.3, .3, .3]),
                    'sampling_rate': 16000},
         'target': 'lug1'},
        {'source': {'path': None,
                    'array': np.array([.4, .4, .4]),
                    'sampling_rate': 16000},
         'target': 'lug2'}]
      
      self.assertNestedAlmostEqual(list(ds), expected)
    
    def test_join_speech_translation_dataset(self):
      yaml_config = '''
      huggingface_load:
          join:
            - path: parquet
              data_files: PATH/audio_mock.parquet
              split: train
            - path: csv
              data_files: PATH/translation_dataset_1.csv
              split: train
      source:
          type: speech
          language: lug
      target:
          type: text
          language: eng
      '''.replace('PATH', self.data_path)
      config = yaml.safe_load(yaml_config)
      ds = dataset.create(config)
        
      expected = [
        {'source': {'path': None,
                    'array': np.array([.1, .1, .1]),
                    'sampling_rate': 16000},
         'target': 'eng1'},
        {'source': {'path': None,
                    'array': np.array([.3, .3, .3]),
                    'sampling_rate': 16000},
         'target': 'eng1'},
        {'source': {'path': None,
                    'array': np.array([.2, .2, .2]),
                    'sampling_rate': 16000},
         'target': 'eng2'},
        {'source': {'path': None,
                    'array': np.array([.4, .4, .4]),
                    'sampling_rate': 16000},
         'target': 'eng2'}]
      
      self.assertNestedAlmostEqual(list(ds), expected)

    def test_speech_to_speech_dataset(self):
      yaml_config = '''
      huggingface_load:
          join:
            - path: parquet
              data_files: PATH/audio_mock.parquet
              split: train
            - path: csv
              data_files: PATH/translation_dataset_1.csv
              split: train
      source:
          type: speech
          language: lug
      target:
          type: speech
          language: ach
      '''.replace('PATH', self.data_path)
      config = yaml.safe_load(yaml_config)
      ds = dataset.create(config)
      
      expected = [
        {'source': {'path': None,
                    'array': np.array([.1, .1, .1]),
                    'sampling_rate': 16000},
         'target': {'path': None,
                    'array': np.array([.8, .8, .8]),
                    'sampling_rate': 16000}},
        {'source': {'path': None,
                    'array': np.array([.3, .3, .3]),
                    'sampling_rate': 16000},
         'target': {'path': None,
                    'array': np.array([.8, .8, .8]),
                    'sampling_rate': 16000}},
      ]

      self.assertNestedAlmostEqual(list(ds), expected)
        
    # TODO: check error is raised if trying to join unsorted
    
if __name__ == '__main__':
    unittest.main()

