import datasets
from datasets import Dataset, Audio, Value, Features
import itertools
import text_preprocessing
import functools

def _ensure_list(x):
    return x if isinstance(x, list) else [x]

def _load_single_huggingface_dataset(load_dataset_params):
    # TODO: If single audio file, reformat to match joined audio+translate.
    ds = datasets.load_dataset(**load_dataset_params)
    if 'train' in ds or 'test' in ds:
        raise ValueError(
            "The dataset split should be specified in config, e.g. "
            "'split: train' or 'split: train+test'. Config provided: "
            f"{load_dataset_params}. Splits found: {list(ds.keys())}."
        )
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
            elif key.startswith('text_'):
                combined_entry[key] = value
            elif key == 'audio':
                language_key = 'audio_' + entry['audio_language']
                combined_entry.setdefault(language_key, []).append(value)
                speaker_id_key = f'audio_{entry["audio_language"]}_speaker_id'
                combined_entry.setdefault(
                    speaker_id_key, []).append(entry['speaker_id'])
        return combined_entry

    # TODO: Work around the sorted() call, which requires everything in memory.
    merged_datasets = sorted(
        itertools.chain(left, right), key=lambda x: x['id'])

    current_id = None
    combined_entry = {}
    for entry in merged_datasets:
        entry_id = entry['id']
        if entry_id != current_id and current_id is not None:
            yield combined_entry
            combined_entry = {}
        current_id = entry_id
        combined_entry['id'] = entry_id
        combined_entry = update_combined_entry(combined_entry, entry)

    if combined_entry:
        yield combined_entry

def _load_huggingface_datasets(config):
    """Retrieve all specified HuggingFace datasets and return as a list."""
    loaded_datasets = []
    if 'huggingface_load' not in config:
        raise ValueError(
            'There should be a `huggingface_load` entry in the dataset config, '
            f'specifying which datasets to download. Got: {config}.'
        )

    load_list = config['huggingface_load']
    for l in _ensure_list(load_list):
        if 'join' in l:
            if not isinstance(l['join'], list) or len(l['join']) != 2:
                raise ValueError(
                    'If a dataset join is specified, then there should be a '
                    f'list of exactly two datasets to be joined. Got: {l}.'
                )
            left = _load_single_huggingface_dataset(l['join'][0])
            right = _load_single_huggingface_dataset(l['join'][1])
            ds = _combine_datasets_generator(left, right)
        else:
            ds = _load_single_huggingface_dataset(l)
        loaded_datasets.append(ds)
    return loaded_datasets

def _matching_items(row, source_target_config):
    """Find which items in a row match the config."""
    matches = []
    speaker_id_filter = source_target_config.get('speaker_id')
    for language in _ensure_list(source_target_config['language']):
        if source_target_config['type'] == 'text':
            if row.get(f'text_{language}'):
                matches.append(
                    {'text': row[f'text_{language}'],
                     'language': language,
                     'origin_dataset': None, # TODO
                    })
        elif source_target_config['type'] == 'speech':
            if row.get('audio_language') == language:
                if speaker_id_filter and row['speaker_id'] != speaker_id_filter:
                    continue
                matches.append(
                    {'audio': row['audio'],
                     'language': row['audio_language'],
                     'speaker_id': row['speaker_id'],
                     'is_studio': row['is_studio'],
                    })
            if f'audio_{language}' in row:
                for audio_example, speaker_id in zip(
                    row[f'audio_{language}'],
                    row[f'audio_{language}_speaker_id']):
                    if speaker_id_filter and speaker_id != speaker_id_filter:
                        continue
                    matches.append({
                        'audio': audio_example,
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
    source_items = _matching_items(row, config['source'])
    target_items = _matching_items(row, config['target'])
    
    for source in source_items:
        for target in target_items:
            example = {
                'source': source.get('text') or source.get('audio'),
                'target': target.get('text') or target.get('audio'),
            }
            
            for k, v in source.items():
                if k not in ['text', 'audio']:
                    example['source_' + k] = v
                    
            for k, v in target.items():
                if k not in ['text', 'audio']:
                    example['target_' + k] = v

            yield example

def _create_generator(config):
    '''Make a generator that yields examples according to dataset spec.'''    
    huggingface_datasets = _load_huggingface_datasets(config)
    for ds in huggingface_datasets:
        for row in ds:
            for match in _matching_pairs(row, config):
                yield match

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

    functions = []
    for f in preprocess_spec:
        # The YAML for each preprocessing operation looks like either 
        # "function_name" or {"function_name": kwargs} 
        if isinstance(f, str):
            function_name = f
            kwargs = {}
        else:
            function_name = list(f.keys())[0]
            kwargs = f[function_name]
                
        # Find the corresponding function definition    
        if config[source_or_target]['type'] == 'text':
            function = getattr(text_preprocessing, function_name)
        else:
            raise NotImplementedError() # TODO audio
                    
        functions.append(
            functools.partial(function, src_or_tgt=source_or_target, **kwargs))
                
    preprocess_fn = _compose(functions)        
    return preprocess_fn
    
def _keep_only_source_and_target(row):
    result = {
        'source': row['source'],
        'target': row['target'],
    }
    return result

def _build_preprocessing_functions(config):
    '''Create functions to process source and target examples.'''
    source_preprocess_fn = _build_source_or_target_preprocess_function(
        config, 'source')    
    target_preprocess_fn = _build_source_or_target_preprocess_function(
        config, 'target')    
    combined_fn = _compose([
        source_preprocess_fn,
        target_preprocess_fn,
        _keep_only_source_and_target,
    ])
    return combined_fn
    
def create(config):
    """
    Create a dataset from the given configuration.

    Args:
      huggingface_load : Dict containing keyword arguments to HuggingFace
          datasets.load_dataset(), or a list of dicts to load multiple
          datasets.
      source: Dict containing source specification, as below.
      target: Dict containing target specification, as below.
      shuffle: Whether to shuffle the data after loading (default False).

    Source and target configuration:
      language: Either an ISO 639-2 language code (e.g. 'eng', 'lug'),
          or a list of codes.
      type: 'text' or 'speech'.
      recording_type: In the case of audio, 'studio', 'natural' or
          'any' (default).
      preprocessing: list of any functions that should be applied at
          load time (done once, same output every epoch).
      preprocessing_on_the_fly: list of any functions that should be applied
          subsequently, on the fly (e.g. augmentation, for different output
          every epoch).

    Returns:
      dataset: A datasets.Dataset object with attributes `source` and `target`.
    """
    # TODO: checks on configuration to make sure it's valid.
    # TODO: make sample rate configurable.
    
    for language in [config[s]['language'] for s in ['source', 'target']]:
        if ',' in language and isinstance(language, str):
            raise ValueError(
                'A list of languages has been specified in config as a string: '
                f'{language}. Change to [{language}] to make it a list.')
        
    text_features = datasets.Features({
        'text': Value('string'),
        'language': Value('string'),
        'origin_dataset': Value('string'),
    })
    
    audio_features = datasets.Features({
        'audio': Audio(sampling_rate=16_000),
        'language': Value('string'),
        'speaker_id': Value('string'),
        'is_studio': Value('bool'),
    })
    
    features = {}
    for source_or_target in ['source', 'target']:
        if config[source_or_target]['type'] == 'text':
            for k, v in text_features.items():
                if k == 'text':
                    features[source_or_target] = v
                else:
                    features[f'{source_or_target}_{k}'] = v
        else:
            for k, v in audio_features.items():
                if k == 'audio':
                    features[source_or_target] = v
                else:
                    features[f'{source_or_target}_{k}'] = v
    
    generator_function = lambda: _create_generator(config)
    ds = datasets.Dataset.from_generator(
        generator_function, features=datasets.Features(features))

    # Apply preprocessing
    preprocessing_fn = _build_preprocessing_functions(config)
    ds.set_transform(preprocessing_fn)    
    return ds

    