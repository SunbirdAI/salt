import math
import datasets
import itertools
import functools
import time
import types
import heapq
import random
import numpy as np
import threading
import concurrent.futures

from . import preprocessing

_PREPROCESS_LIST_HELP_STRING = '''
Preprocessing operations should be specified as a list in yaml, for example:

preprocessing:
    - first_operation:
        params
    - second_operation

and not like this (without dashes):

preprocessing:
    first_operation:
        params
    second_operation 
'''

def _ensure_list(x):
    return x if isinstance(x, list) else [x]

def _common_voice_to_SALT(batch, language):
    '''Remap a Common Voice format batch to SALT format.'''
    # Process the whole batch at once
    batch_size = len(batch['sentence'])
    # Transform data
    batch['id'] = [-1] * batch_size # TODO: create sequential IDs
    batch['sample_rate'] = [example['sampling_rate'] for example in batch['audio']]
    batch['audio'] = [example['array'] for example in batch['audio']]
    batch['text'] = batch['sentence']
    batch['language'] = [language] * batch_size
    batch['speaker_id'] = [0] * batch_size
    batch['is_studio'] = [False] * batch_size
    return batch

def _google_fleurs_to_SALT(batch, language):
    '''Remap a Google FLEURS format batch to SALT format.'''
    # Process the whole batch at once
    batch_size = len(batch['transcription'])
    # Transform data
    batch['id'] = [-1] * batch_size # TODO: create sequential IDs
    batch['sample_rate'] = [example['sampling_rate'] for example in batch['audio']]
    batch['audio'] = [example['array'] for example in batch['audio']]
    batch['text'] = batch['raw_transcription']
    batch['language'] = [language] * batch_size
    batch['speaker_id'] = [0] * batch_size
    batch['is_studio'] = [False] * batch_size
    return batch    

def _add_speaker_id_studio_if_not_present(sample):
    if 'speaker_id' not in sample:
        sample['speaker_id'] = 0
    if 'is_studio' not in sample:
        sample['is_studio'] = False
    return sample

def _load_single_huggingface_dataset(load_dataset_params):
    ds = datasets.load_dataset(**load_dataset_params)
    if isinstance(ds, datasets.DatasetDict):
        split_names = list(ds.data.keys())
        # If the split wasn't specified, but there's only one, then just go
        # ahead and load that one.
        if len(split_names) == 1:
            ds = ds[split_names[0]]
        else:
            raise ValueError(
                "The dataset split should be specified in config, e.g. "
                "'split: train' or 'split: train+test'. Config provided: "
                f"{load_dataset_params}. Splits found: {list(ds.keys())}."
            )
        
    remap_names = {
        'audio_language': 'language',
    }
    
    for from_name, to_name in remap_names.items():
        if from_name in ds.features and not to_name in ds.features:
            ds = ds.rename_column(from_name, to_name)
      
    # If this is a Common Voice dataset, then remap it to SALT format.
    COMMON_VOICE_LANGUAGE_MAPPING = {
        'lg': 'lug',
        'sw': 'swa',
        'rw': 'kin',
        'ig': 'ibo',
        'yo': 'yor',
        'am': 'amh',
        'ti': 'tir',
        'ha': 'hau',
        'zu': 'zul', 
        'nso': 'nso',
    }
    if load_dataset_params['path'].startswith('mozilla-foundation/common_voice'):
        if load_dataset_params['name'] in COMMON_VOICE_LANGUAGE_MAPPING:
            language = COMMON_VOICE_LANGUAGE_MAPPING[load_dataset_params['name']]
        else:
            language = load_dataset_params['name']
            available_configs = ', '.join(
                [k for k in COMMON_VOICE_LANGUAGE_MAPPING.keys()])
            raise Warning(
                'Not sure how to map the Common Voice subset '
               f'{load_dataset_params["name"]} to a SALT language code. '
               f'Available options are: {available_configs}.')
        ds.set_transform(
            lambda x: _common_voice_to_SALT(x, language))

    # If this is a Google FLEURS dataset, then remap it to SALT format.
    elif load_dataset_params['path'] == 'google/fleurs':
        if load_dataset_params['name'] == 'lg_ug':
            language = 'lug'
        elif load_dataset_params['name'] == 'sw_ke':
            language = 'swa'
        else:
            language = load_dataset_params['name']
            raise Warning(
                'Not sure how to map the FLEURS subset '
               f'{load_dataset_params["name"]} to a SALT language code.')
        ds.set_transform(
            lambda x: _google_fleurs_to_SALT(x, language))

    return ds

def _combine_datasets_generator(left, right):
    """
    A generator that yields combined text and audio data from two sorted
    datasets based on unique IDs. It expects 'left' and 'right' to be sorted
    by 'id'.
    """
    def update_combined_entry(combined_entry, entry):
        for key, value in entry.items():
            if key == 'id':
                continue
            elif key.endswith('_text'):
                combined_entry[key] = value
            elif key == 'audio':
                language_key = 'audio_' + entry['language']
                combined_entry.setdefault(language_key, []).append(value)
                sample_rate_key = f'audio_{entry["language"]}_sample_rate'
                combined_entry.setdefault(
                    sample_rate_key, []).append(entry['sample_rate'])                
                speaker_id_key = f'audio_{entry["language"]}_speaker_id'
                combined_entry.setdefault(
                    speaker_id_key, []).append(entry['speaker_id'])
        return combined_entry

    merged_datasets = heapq.merge(left, right, key=lambda x: int(x['id']))
    
    current_id = None
    combined_entry = {}
    for entry in merged_datasets:
        entry_id = int(entry['id'])
        if entry_id != current_id and current_id is not None:
            if int(entry_id) < int(current_id):
                raise ValueError(
                    'To join two datasets based on the `id` field, the ids '
                    'must be in numerical sorted order within both datasets. '
                    f'Found id {entry_id} after {current_id}.')
            yield combined_entry
            combined_entry = {}
        current_id = entry_id
        combined_entry['id'] = entry_id
        combined_entry = update_combined_entry(combined_entry, entry)
    if combined_entry:
        yield combined_entry

def _dataset_id_from_config(load_params):
    tag = [load_params.get('path')]
    if 'name' in load_params:
        tag.append(load_params.get('name'))
    if 'data_files' in load_params:
        if isinstance(load_params['data_files'], dict):
            tag.extend(''.join(
                "{!r}".format(val)
                for (key,val) in load_params['data_files'].items()))
        else:
            tag.extend(_ensure_list(load_params['data_files']))
    return '_'.join(tag)

def _load_huggingface_datasets(config):
    """Retrieve all specified HuggingFace datasets and return as a list."""
    loaded_datasets = []
    if 'huggingface_load' not in config:
        raise ValueError(
            'There should be a `huggingface_load` entry in the dataset config, '
            f'specifying which datasets to download. Got: {config}.'
        )

    load_list = config['huggingface_load']

    # Optionally pre-download everything at once
    if config.get('download_datasets_in_parallel'):
        threads = []
        for l in _ensure_list(load_list):
            if 'join' in l:
                for i in (0, 1):
                    thread = threading.Thread(
                        target=_load_single_huggingface_dataset, args=(l['join'][i],))
                    threads.append(thread)
            else:
                thread = threading.Thread(
                    target=_load_single_huggingface_dataset, args=(l,))
                threads.append(thread)
                thread.start()
        for thread in threads:
            thread.join()

    for l in _ensure_list(load_list):
        if 'join' in l:
            if not isinstance(l['join'], list) or len(l['join']) != 2:
                raise ValueError(
                    'If a dataset join is specified, then there should be a '
                    f'list of exactly two datasets to be joined. Got: {l}.'
                )
            left = _load_single_huggingface_dataset(l['join'][0])
            right = _load_single_huggingface_dataset(l['join'][1])
            
            generator_function = lambda: _combine_datasets_generator(left,
                                                                     right)
            ds = datasets.IterableDataset.from_generator(generator_function)
            dataset_id = (_dataset_id_from_config(l['join'][0]) + ',' +
                          _dataset_id_from_config(l['join'][1]))
        else:
            ds = _load_single_huggingface_dataset(l)
            dataset_id = _dataset_id_from_config(l)
        loaded_datasets.append([ds, dataset_id])
        
    if config.get('shuffle'):
        loaded_datasets = [[ds[0].shuffle(), ds[1]] for ds in loaded_datasets]
        
    return loaded_datasets

def _get_audio_from_row(row):
    if 'audio' not in row:
        raise ValueError(
            'Trying to read audio, but there is no `audio` feature.')

    if isinstance(row['audio'], dict):
        audio_array = row['audio']['array']
        sample_rate = row['audio']['sampling_rate']
    else:
        audio_array = row['audio']
        sample_rate = row['sample_rate']
    return audio_array, sample_rate

def _matching_items(row, source_target_config, source_or_target):
    """Find which items in a row match the config."""
    matches = []
    speaker_id_filter = source_target_config.get('speaker_id')
    # TODO: filter based on recording type (natural/studio/any)
    for language in _ensure_list(source_target_config['language']):  
        if source_target_config['type'] == 'text':
            if source_or_target == 'target' and row.get(f'{language}_target_text'):
                matches.append(
                    {'text': row[f'{language}_target_text'],
                     'language': language,
                     'origin_dataset': row['origin_dataset'],
                    })
            if source_or_target == 'source' and row.get(f'{language}_source_text'):
                matches.append(
                    {'text': row[f'{language}_source_text'],
                     'language': language,
                     'origin_dataset': row['origin_dataset'],
                    })
            elif row.get(f'{language}_text'):
                matches.append(
                    {'text': row[f'{language}_text'],
                     'language': language,
                     'origin_dataset': row['origin_dataset'],
                    })
        elif source_target_config['type'] == 'speech':
            if row.get('language') == language:    
                if speaker_id_filter and row['speaker_id'] != speaker_id_filter:
                    continue
                audio_array, sample_rate = _get_audio_from_row(row)
                matches.append(
                    {'audio': audio_array,
                     'sample_rate': sample_rate,
                     'language': row.get('language') or row.get('audio_language'),
                     'speaker_id': row.get('speaker_id'),
                     'is_studio': row.get('is_studio'),
                    })
            if row.get(f'audio_{language}'):
                for audio_example, sample_rate, speaker_id in zip(
                    row[f'audio_{language}'],
                    row[f'audio_{language}_sample_rate'],
                    row[f'audio_{language}_speaker_id']):
                    if speaker_id_filter and speaker_id != speaker_id_filter:
                        continue
                    matches.append({
                        'audio': audio_example,
                        'sample_rate': sample_rate,
                        'language': language,
                        'speaker_id': speaker_id,
                        'is_studio': None,  # TODO
                    })
        else:
            raise ValueError(
                'Unknown source/target type. Should be one of '
                f'"speech" or "text", got: {source_target_config}.'
            )
    return matches

    
def _matching_pairs(row, config):
    """Find all source/target pairs that match the configuration."""
    
    source_items = _matching_items(row, config['source'], 'source')
    target_items = _matching_items(row, config['target'], 'target')

    remove_reverse_duplicates = config.get('no_reverse_duplicate_examples')
    if remove_reverse_duplicates:
        random.shuffle(source_items)
        random.shuffle(target_items)

    processed_language_pairs = set()

    for source in source_items:
        for target in target_items:
            example = {}
            for k, v in source.items():
                if k not in ['text', 'audio']:
                    example['source.' + k] = v
                else:
                    example['source'] = v
                    
            for k, v in target.items():
                if k not in ['text', 'audio']:
                    example['target.' + k] = v
                else:
                    example['target'] = v
                    
            matching_src_tgt_languages = (
                example['source.language'] == example['target.language'])
                    
            if (matching_src_tgt_languages and
                not config.get('allow_same_src_and_tgt_language', True)):
                continue
            
            required_language = config.get('src_or_tgt_languages_must_contain')
            if required_language:
                if not (example['source.language'] == required_language or
                        example['target.language'] == required_language):
                    continue

            lang_pair = frozenset([example['source.language'], example['target.language']])

            if remove_reverse_duplicates and lang_pair in processed_language_pairs:
                continue

            processed_language_pairs.add(lang_pair)
            yield example
    

def _create_generator(config, verbose=False):
    '''Make a generator that yields examples according to dataset spec.'''    
    huggingface_datasets = _load_huggingface_datasets(config)

    if verbose:
        total_row_count = 0
        for ds, id in huggingface_datasets:
            row_count = len(ds)
            total_row_count += row_count
            print(f'{id}: {row_count} rows')
        print(f'Total rows: {total_row_count}')
    
    def _yield_matches(batch, config, dataset_id):
        keys = list(batch.keys())
        rows = [
            {k: batch[k][i] for k in keys}
             for i in range(len(batch[keys[0]]))
        ]
        for row in rows:  
            # The audio SALT datasets are in a slightly different format
            # to the translation data, each row having a 'text' and 'language'
            # field.
            if 'audio' in row and 'text' in row:
                row[row['language'] + '_text'] = row['text']
                del row['text']
            for match in _matching_pairs(
                row | {'origin_dataset': dataset_id}, config):
                yield match
                
    # PyArrow data should be read in batches for speed.
    PYARROW_BATCH_SIZE = 10
    num_workers = config.get('num_workers', 4)
            
    if config.get('shuffle') and len(huggingface_datasets) > 1:
        # If there are multiple datasets concatenated and 'shuffle' is
        # specified, then we want to randomly interleave them.
        iterators = [d[0].iter(batch_size=PYARROW_BATCH_SIZE)
                     for d in huggingface_datasets]
        iterator_order = []
        for i in range(len(huggingface_datasets)):
            num_batches = math.ceil(
                len(huggingface_datasets[i][0]) / PYARROW_BATCH_SIZE)
            iterator_order.extend([i] * num_batches)
        permutation = np.random.permutation(len(iterator_order))
        iterator_order = np.array(iterator_order)[permutation]
        def process_batch(args):
            batch, config, dataset_id = args
            return list(_yield_matches(batch, config, dataset_id))
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_list = []
            for iterator_id in iterator_order:
                try:
                    batch = next(iterators[iterator_id])
                except Exception as e:
                    print('Error reading from ' + huggingface_datasets[iterator_id][1])
                    raise
                future = executor.submit(process_batch, (batch, config, huggingface_datasets[iterator_id][1]))
                future_list.append(future)
            for future in future_list:
                for match in future.result():
                    yield match
    elif config.get('shuffle'):
        # Single dataset, shuffle is True: parallelize batches, order doesn't matter
        def process_batch(args):
            batch, config, dataset_id = args
            return list(_yield_matches(batch, config, dataset_id))
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for ds, dataset_id in huggingface_datasets:  
                for batch in ds.iter(batch_size=PYARROW_BATCH_SIZE): 
                    futures.append(executor.submit(process_batch, (batch, config, dataset_id)))
            for future in concurrent.futures.as_completed(futures):
                for match in future.result():
                    yield match
    else:
        # No shuffle: preserve strict order, process sequentially
        for ds, dataset_id in huggingface_datasets:  
            for batch in ds.iter(batch_size=PYARROW_BATCH_SIZE): 
                yield from _yield_matches(batch, config, dataset_id)
                
def _compose(functions):
    def inner(arg):
        for f in functions:
            arg = f(arg)
        return arg
    return inner
                 
def _build_source_or_target_preprocess_function(config, source_or_target):
    '''Compose the specified preprocessing ops into one function object.'''
    preprocess_spec = config[source_or_target].get('preprocessing')
    
    # If nothing is specified, just return the identify function.
    if not preprocess_spec:
        return lambda x: x
    
    if isinstance(preprocess_spec, dict):
        # Easy mistake to make: specifying preprocessing ops as a dict (which
        # is not appropriate as it has no ordering). Alert the user.
        print(_PREPROCESS_LIST_HELP_STRING)
        raise ValueError(
            'Error found in preprocessing specification: ',
            preprocess_spec
        )
        
    functions = []
    for f in preprocess_spec:
        # The YAML for each preprocessing operation looks like either 
        # "function_name" or {"function_name": kwargs} 
        if isinstance(f, str):
            function_name = f
            kwargs = {}
        else:
            function_name = list(f.keys())[0]
            kwargs = f[function_name] or {}
                
        available_function_names = [
            name for name in dir(preprocessing)
            if callable(getattr(preprocessing, name))]
        
        if function_name not in available_function_names:
            raise NameError(
                f'Preprocessing function \'{function_name}\' couldn\'t be '
                f'loaded. Available {config[source_or_target]["type"]} '
                f'preprocessing functions are: {available_function_names}.')
            
        function = getattr(preprocessing, function_name)            
        functions.append(
            functools.partial(function, src_or_tgt=source_or_target, **kwargs))
                
    preprocess_fn = _compose(functions)        
    return preprocess_fn
    
def _build_preprocessing_functions(config):
    '''Create functions to process source and target examples.'''
    source_preprocess_fn = _build_source_or_target_preprocess_function(
        config, 'source')   
    target_preprocess_fn = _build_source_or_target_preprocess_function(
        config, 'target')    
    combined_fn = _compose([
        source_preprocess_fn,
        target_preprocess_fn,
    ])
    return combined_fn
    
def create(config, verbose=False):
    """
    Create a dataset from the given configuration.

    Args:
      huggingface_load : Dict containing keyword arguments to HuggingFace
          datasets.load_dataset(), or a list of dicts to load multiple
          datasets. The dataset should be in SALT format, as per
          hf.co/datasets/sunbird/salt. Common Voice is also supported, if
          loaded from the `mozilla-foundation/common_voice_13_0` repo.
      source: Dict containing source specification, as below.
      target: Dict containing target specification, as below.
      shuffle: Whether to shuffle the data after loading (default False).

    Source and target configuration:
      language: Either an ISO 639-2 language code (e.g. 'eng', 'lug'),
          or a list of codes.
      type: 'text' or 'speech'.
      preprocessing: list of any functions that should be applied to transform
          the data.

    Returns:
      dataset: A datasets.Dataset object with attributes `source`, `target`,
          `source.language` and `target.language`.
          
    See notebooks/Leb test.ipynb for example usage.
    """
    # TODO: checks on configuration to make sure it's valid.
   
    # Multiple source or target languages can be specified in the yaml config
    # e.g. with "language: [lug, ach]". An easy mistake is to write
    # "language: lug, ach" instead , which gets converted to a string and not
    # a list, so check for that and alert the user.
    for language in [config[s]['language'] for s in ['source', 'target']]:
        if ',' in language and isinstance(language, str):
            raise ValueError(
                'A list of languages has been specified in config as a string: '
                f'{language}. Change to [{language}] to make it a list.')
            
    generator_function = lambda: _create_generator(config, verbose=verbose)
    ds = datasets.IterableDataset.from_generator(generator_function)
    
    # The individual datasets are already shuffled as needed, but do a little
    # more so that consecutive samples are from different batches.
    if config.get('shuffle'):
        ds = ds.shuffle(buffer_size=50)

    # Apply preprocessing
    preprocessing_fn = _build_preprocessing_functions(config)
    ds = ds.map(preprocessing_fn, batched=True, batch_size=10)
    
    if not config.get('keep_metadata_features'):
        ds = ds.select_columns(
            ['source', 'target', 'source.language', 'target.language'])
    return ds