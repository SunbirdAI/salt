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
           
        translate_data_a = {
            'id': range(1000),
            'lug_text': ['a1'] * 1000,
            'eng_text': ['a2'] * 1000,
        }
        
        translate_data_b = {
            'id': range(1000),
            'lug_text': ['b1'] * 1000,
            'eng_text': ['b2'] * 1000,
        }
                
        audio_metadata = {
            'id': [1, 1, 1, 1, 2, 2, 2, 3,],
            'audio': [
                [.1, .1, .1],
                [.3, .3, .3],
                [.5, .5, .5],
                [.8, .8, .8],
                [.2, .2, .2],
                [.4, .4, .4],
                [.6, .6, .6],
                [.7, .7, .7],
            ],
            'sample_rate': [16_000] * 8,
            'text': [
              'lug1', 'lug1', 'eng1', 'ach1', 'lug2',  'lug2',  'eng2', 'eng3'],
            'audio_language': [
              'lug', 'lug', 'eng', 'ach', 'lug', 'lug', 'eng', 'eng'],
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
        
        # Phrase ID 3 comes before IDs 1 and 2
        audio_metadata_unsorted = {
            'id': [3, 1, 1, 1, 1, 2, 2, 2],
            'audio': [
                [.7, .7, .7],
                [.1, .1, .1],
                [.3, .3, .3],
                [.5, .5, .5],
                [.8, .8, .8],
                [.2, .2, .2],
                [.4, .4, .4],
                [.6, .6, .6],
            ],
            'sample_rate': [16_000] * 8,
            'text': [
              'eng3', 'lug1', 'lug1', 'eng1', 'ach1', 'lug2',  'lug2', 'eng2'],
            'audio_language': [
              'eng', 'lug', 'lug', 'eng', 'ach', 'lug', 'lug', 'eng'],
            'is_studio': [
              False, False, True, True, False, False, False, False],
            'speaker_id': [
                'eng-001',
                'lug-001',
                'lug-studio-1',
                'eng-001',
                'ach-001',
                'lug-002',
                'lug-studio-1',
                'eng-002',
            ]   
        }

        audio_object_metadata = {
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
              'lug', 'lug', 'eng', 'ach', 'lug', 'lug', 'eng', 'eng'],
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
        
        temp_csv_path = f'{self.data_path}/translation_dataset_1.csv'
        pd.DataFrame(translate_data_1).to_csv(temp_csv_path, index=False)    

        temp_csv_path = f'{self.data_path}/translation_dataset_2.csv'
        pd.DataFrame(translate_data_2).to_csv(temp_csv_path, index=False)
        
        temp_csv_path = f'{self.data_path}/translation_missing_value.csv'
        pd.DataFrame(translate_data_missing_value).to_csv(
          temp_csv_path, index=False)
        
        temp_csv_path = f'{self.data_path}/translation_dataset_a1k.csv'
        pd.DataFrame(translate_data_a).to_csv(temp_csv_path, index=False)
        
        temp_csv_path = f'{self.data_path}/translation_dataset_b1k.csv'
        pd.DataFrame(translate_data_b).to_csv(temp_csv_path, index=False)
   
        audio_dataset = datasets.Dataset.from_dict(audio_metadata)
        audio_dataset.to_parquet(f'{self.data_path}/audio_mock.parquet')

        audio_dataset_unsorted = datasets.Dataset.from_dict(
            audio_metadata_unsorted)
        audio_dataset_unsorted.to_parquet(
            f'{self.data_path}/audio_mock_unsorted.parquet')

        audio_object_dataset = datasets.Dataset.from_dict(audio_object_metadata).cast_column(
          'audio', datasets.Audio(sampling_rate=16000))
        audio_object_dataset.to_parquet(f'{self.data_path}/audio_object_mock.parquet')

        datasets.disable_progress_bar()
        # HuggingFace datasets gives a ResourceWarning with temp files
        warnings.simplefilter("ignore", ResourceWarning)
        
    def tearDown(self):
        self.temp_dir.cleanup()
        warnings.simplefilter("default", ResourceWarning)
      
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
            {'source': 'two one lug1 three',
             'target': 'eng1 four',
             'source.language': 'lug',
             'target.language': 'eng',
            },
            {'source': 'two one lug2 three',
             'target': 'eng2 four',
             'source.language': 'lug',
             'target.language': 'eng',
            },
            {'source': 'two one lug3 three',
             'target': 'eng3 four',
             'source.language': 'lug',
             'target.language': 'eng',
            },
        ]

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
         'target': np.array([.3, .3, .3]),
         'source.language': 'lug',
         'target.language': 'lug',
        },
        {'source': 'lug2',
         'target': np.array([.4, .4, .4]),
         'source.language': 'lug',
         'target.language': 'lug',
        },
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
            {'source': 'lug1',
             'target': 'eng1',
             'source.language': 'lug',
             'target.language': 'eng',             
            },
            {'source': 'lug2',
             'target': 'eng2',
             'source.language': 'lug',
             'target.language': 'eng',             
            },
            {'source': 'lug3',
             'target': 'eng3',
             'source.language': 'lug',
             'target.language': 'eng',            
            }
        ]

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
            {'source': 'lug1',
             'target': 'eng1',
             'source.language': 'lug',
             'target.language': 'eng',             
            },
            {'source': 'lug2',
             'target': 'eng2',
             'source.language': 'lug',
             'target.language': 'eng',             
            },
            {'source': 'lug3',
             'target': 'eng3',
             'source.language': 'lug',
             'target.language': 'eng',             
            },
            {'source': 'ach1',
             'target': 'eng1',
             'source.language': 'ach',
             'target.language': 'eng',             
            },
            {'source': 'ach2',
             'target': 'eng2',
             'source.language': 'ach',
             'target.language': 'eng',             
            },
            {'source': 'ach3',
             'target': 'eng3',
             'source.language': 'ach',
             'target.language': 'eng',             
            },
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
            {'source': 'lug1',
             'target': 'eng1',
             'source.language': 'lug',
             'target.language': 'eng',            
            },
            {'source': 'lug2',
             'target': 'eng2',
             'source.language': 'lug',
             'target.language': 'eng',             
            },
            {'source': 'lug3',
             'target': 'eng3',
             'source.language': 'lug',
             'target.language': 'eng',             
            },
            {'source': 'lug4',
             'target': 'eng4',
             'source.language': 'lug',
             'target.language': 'eng',             
            },
            {'source': 'lug5',
             'target': 'eng5',
             'source.language': 'lug',
             'target.language': 'eng',             
            }
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
            {'source': 'lug1',
             'target': 'eng1',
             'source.language': 'lug',
             'target.language': 'eng',
            },
            {'source': 'lug3',
             'target': 'eng3',
             'source.language': 'lug',
             'target.language': 'eng',            
            },
            {'source': 'lug4',
             'target': 'eng4',
             'source.language': 'lug',
             'target.language': 'eng',            
            },
            {'source': 'lug5',
             'target': 'eng5',
             'source.language': 'lug',
             'target.language': 'eng',            
            }
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
        {'source': np.array([.1, .1, .1]),
         'target': 'lug1',
         'source.language': 'lug',
         'target.language': 'lug',        
        },
        {'source': np.array([.3, .3, .3]),
         'target': 'lug1',
         'source.language': 'lug',
         'target.language': 'lug',          
        },
        {'source': np.array([.2, .2, .2]),
         'target': 'lug2',
         'source.language': 'lug',
         'target.language': 'lug',          
        },
        {'source': np.array([.4, .4, .4]),
         'target': 'lug2',
         'source.language': 'lug',
         'target.language': 'lug',          
        }
      ]
      
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
        {'source': np.array([.1, .1, .1]),
         'target': 'eng1',
         'source.language': 'lug',
         'target.language': 'eng',          
        },
        {'source': np.array([.3, .3, .3]),
         'target': 'eng1',
         'source.language': 'lug',
         'target.language': 'eng',        
        },
        {'source': np.array([.2, .2, .2]),
         'target': 'eng2',
         'source.language': 'lug',
         'target.language': 'eng',        
        },
        {'source': np.array([.4, .4, .4]),
         'target': 'eng2',
         'source.language': 'lug',
         'target.language': 'eng',        
        }
      ]
      
      self.assertNestedAlmostEqual(list(ds), expected)
    
    
    def test_join_speech_translation_dataset_with_resample(self):
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
          preprocessing:
            - set_sample_rate:
                rate: 32000
      target:
          type: text
          language: eng
      '''.replace('PATH', self.data_path)
      config = yaml.safe_load(yaml_config)
      ds = dataset.create(config)
                
      expected = [
        {'source': np.array([.1, .1, .1]),
         'target': 'eng1',
         'source.language': 'lug',
         'target.language': 'eng',        
        },
        {'source': np.array([.3, .3, .3]),
         'target': 'eng1',
         'source.language': 'lug',
         'target.language': 'eng',        
        },
        {'source': np.array([.2, .2, .2]),
         'target': 'eng2',
         'source.language': 'lug',
         'target.language': 'eng',        
        },
        {'source': np.array([.4, .4, .4]),
         'target': 'eng2',
         'source.language': 'lug',
         'target.language': 'eng',        
        }
      ]
    
      result = list(ds)
      for i in range(4):
          self.assertEqual(len(result[i]['source']), 6)
    

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
        {'source': np.array([.1, .1, .1]),
         'target': np.array([.8, .8, .8]),
         'source.language': 'lug',
         'target.language': 'ach',        
        },
        {'source': np.array([.3, .3, .3]),
         'target': np.array([.8, .8, .8]),
         'source.language': 'lug',
         'target.language': 'ach',        
        },
      ]

      self.assertNestedAlmostEqual(list(ds), expected)
        

    def test_audio_object_dataset(self):
      yaml_config = '''
      huggingface_load:
          path: parquet
          data_files: PATH/audio_object_mock.parquet
          split: train
      source:
          type: speech
          language: ach
      target:
          type: text
          language: ach
      '''.replace('PATH', self.data_path)
      config = yaml.safe_load(yaml_config)
      ds = dataset.create(config)
      
      expected = [
        {'source': np.array([.8, .8, .8]),
         'target': 'ach1',
         'source.language': 'ach',
         'target.language': 'ach',        
        },
      ]

      self.assertNestedAlmostEqual(list(ds), expected)

    def test_join_unsorted_raises_exception(self):
      def try_creating_unsorted():  
          yaml_config = '''
          huggingface_load:
              join:
                - path: parquet
                  data_files: PATH/audio_mock_unsorted.parquet
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
          list(ds)
      self.assertRaises(ValueError, try_creating_unsorted) 


    def test_two_datasets_shuffled(self):        
        yaml_config = '''
        huggingface_load:
          - path: csv
            data_files: PATH/translation_dataset_a1k.csv
            split: train
          - path: csv
            data_files: PATH/translation_dataset_b1k.csv
            split: train
        source:
            type: text
            language: lug
        target:
            type: text
            language: eng
        shuffle: True
        '''.replace('PATH', self.data_path)

        config = yaml.safe_load(yaml_config)
        ds = dataset.create(config)
        result = list(ds)
        # Test the As and Bs are mixed up - there should be two different values
        # in the first 1k rows.
        self.assertEqual(len(set([row['source'] for row in result[:1000]])), 2)

        yaml_config = '''
        huggingface_load:
          - path: csv
            data_files: PATH/translation_dataset_a1k.csv
            split: train
          - path: csv
            data_files: PATH/translation_dataset_b1k.csv
            split: train
        source:
            type: text
            language: lug
        target:
            type: text
            language: eng
        shuffle: False
        '''.replace('PATH', self.data_path)

        config = yaml.safe_load(yaml_config)
        ds = dataset.create(config)
        result = list(ds)
        # Without shuffling it should only be the value from dataset A
        self.assertEqual(len(set([row['source'] for row in result[:1000]])), 1)   
        
    def test_unspecified_split(self):
        # If there's more than one split in the dataset, but the split
        # wasn't specified, then an error should be raised.
        def try_unspecified_split():
            yaml_config = '''
            huggingface_load:
              - path: csv
                data_files: 
                    train: PATH/translation_dataset_1.csv
                    test: PATH/translation_dataset_1.csv
            source:
                type: text
                language: lug
            target:
                type: text
                language: eng
            shuffle: True
            '''.replace('PATH', self.data_path)

            config = yaml.safe_load(yaml_config)
            ds = list(dataset.create(config))

        self.assertRaises(ValueError, try_unspecified_split)
        
        # If split is not specified, but there's only one split in
        # the dataset, then that one should be loaded.
        yaml_config = '''
        huggingface_load:
          - path: csv
            data_files: 
                test: PATH/translation_dataset_2.csv
        source:
            type: text
            language: lug
        target:
            type: text
            language: eng
        shuffle: True
        '''.replace('PATH', self.data_path)

        config = yaml.safe_load(yaml_config)
        ds = list(dataset.create(config))        
        
        self.assertEqual(len(list(ds)), 2)
        
if __name__ == '__main__':
    unittest.main()

