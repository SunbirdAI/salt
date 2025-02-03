'''
Preprocessing functions for text and audio records.

The input to each function includes:
- `r`, the record to be preprocessed
- `src_or_target`: a string, either 'source' or 'target', specifying which part
    of the record to modify.
- Any other keyword arguments defined in the config.

Typical format of a record:

{
  'source': 'Some text',
  'source.language': 'eng',
  'source.origin_dataset': 'salt',
  'target': [0.0, 0.0, ...],
  'target.sample_rate': 16000,
  'target.language': 'lug',
  'target.is_studio': False,
}

The keys depend on whether the source and target are text or audio respectively.

If the @single_batch_entry decorator is removed, then a whole batch is passed
to each function at once. The structure of the input is then e.g.:

{
  'source': ['Text of entry 1', 'Text of entry 2', ...],
  'source.language': ['eng', 'eng', ...]
  ...
}
'''

import re
import copy
import string
import random
import cleantext
import functools
from functools import lru_cache
import numpy as np
import librosa
import datasets
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.audio as naa

from .utils import single_batch_entry


@single_batch_entry
def random_case(r, src_or_tgt, p_all_lower_case=0.4, p_all_upper_case=0.03):
    '''Augment text to be all lower case or all caps.'''
    if np.random.random() < p_all_upper_case:
        r['source'] = r['source'].upper()
        r['target'] = r['target'].upper()
    # Lower case takes precedence
    if np.random.random() < p_all_lower_case:
        r['source'] = r['source'].lower()
        r['target'] = r['target'].lower()
    return r

@single_batch_entry
def prefix_dataset_tag(r, src_or_tgt, tags=None):
    '''If the origin dataset name matches a string, prefix a tag to the text.'''
    for origin, tag in tags.items():
        if origin in r[f'{src_or_tgt}.origin_dataset']:
            r[src_or_tgt] = tag + ' ' + r[src_or_tgt]
    return r

@single_batch_entry
def prefix_target_language(r, src_or_tgt):
    r[src_or_tgt] = f'>>{r["target.language"]}<< ' + r[src_or_tgt]
    return r

@single_batch_entry
def match_target_sentence_format_to_source(r, src_or_tgt):
    '''Match the sentence formatting of the target text to the source text.
    
    Sets the capitalisation of the first letter in the target text, and
    the presence of a trailing full stop, to match whatever is in the source
    text.'''
    if r['source'][0].isupper():
        r['target'] = r['target'][0].upper() + r['target'][1:]
    else:
        r['target'] = r['target'][0].lower() + r['target'][1:]
        
    source_has_full_stop = (r['source'][-1] == '.')
    target_has_full_stop = (r['target'][-1] == '.')
    
    if source_has_full_stop and not target_has_full_stop:
        r['target'] = r['target'] + '.' 
        
    if target_has_full_stop and not source_has_full_stop:
        r['target'] = r['target'][:-1]
        
    return r

@single_batch_entry
def ensure_text_ends_with_punctuation(r, src_or_tgt):
    '''Add a full stop to the end of text, if it doesn't end with punctuation.'''
    punct = list(string.punctuation)
    input_string = r[src_or_tgt]
    if len(input_string):
        if input_string[-1] not in punct:
            input_string += '.'
        r[src_or_tgt] = input_string
    return r

@single_batch_entry
def clean_text(r, src_or_tgt, **clean_text_args):
    r[src_or_tgt] = cleantext.clean(
        r[src_or_tgt], to_ascii=False, lower=False, **clean_text_args)
    return r

@single_batch_entry
def augment_characters(r, src_or_tgt, avg_character_error_rate = 0.03):
    # Define character set for random insertions
    chars = 'abcdefghijklmnopqrstuvwxyz'

    input_string = r[src_or_tgt]
    lam = len(input_string) * avg_character_error_rate
    target_errors = np.random.poisson(lam=lam)
    errors = 0
    
    # Create a list from the input string for easier modifications
    str_list = list(input_string)
    
    while errors < target_errors:
        # Randomly choose deletion, insertion, modification, or duplication
        operation = random.choice(['delete', 'insert', 'modify', 'duplicate'])
        position = random.randint(0, len(str_list) - 1)

        if operation == 'delete' and len(str_list) > 1:
            # Delete a random character
            del str_list[position]
            errors += 1
        elif operation == 'insert':
            # Insert a random character
            str_list.insert(position, random.choice(chars))
            errors += 1
        elif operation == 'modify':
            # Modify a random character
            str_list[position] = random.choice(chars)
            errors += 1
        elif operation == 'duplicate' and len(str_list) >= 2:
            # Duplicate a random character
            str_list.insert(position, str_list[position])
            errors += 1

    # Join the list back into a string and return
    modified_string = ''.join(str_list)
    r[src_or_tgt] = modified_string    
    return r

def augment_words(r, src_or_tgt, **word_augmentation_params):
    word_augmenter = naw.RandomWordAug(**word_augmentation_params)
    r[src_or_tgt] = word_augmenter.augment(r[src_or_tgt])
    return r

@single_batch_entry
def augment_audio_speed(r, src_or_tgt, p=0.5, low=0.95, high=1.15):
    '''Change the speed of an audio sample randomly.
    
    Args:
        r: dictionary containing fields of a single dataset row.
        src_or_tgt: str, key such that r[src_or_tgt] contains the audio array
            to be augmented.
        p: float, probability that the augmentation is applied, in range (0,1)
        low: float, lower limit for speed change. Default is 0.9 (i.e., slow
            down the speed).
        high: float, upper limit for speed change. Default is 1.5 (i.e.,
            increase the speed).
    '''
    x = r[src_or_tgt]
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    # Do nothing for empty inputs
    if not len(x):
        return r
    
    if np.random.random() < p:
        speed_factor = np.random.uniform(low=low, high=high)
        x_with_speed_change = librosa.effects.time_stretch(x, rate=speed_factor)
        r[src_or_tgt] = x_with_speed_change
        
    return r
    

class NoiseAugmenter:
    """Class to handle noise augmentation with lazy loading of noise datasets."""
    
    _instance = None
    _noise_dataset = None
    _noise_repo_config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NoiseAugmenter, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_noise_dataset(cls, noise_audio_repo):
        """Lazy load the noise dataset only when needed."""
        # Convert dict to frozen set of items for hashability
        config_items = frozenset(noise_audio_repo.items())
        
        # If config changed, invalidate cached dataset
        if cls._noise_repo_config != config_items:
            cls._noise_dataset = None
            cls._noise_repo_config = config_items
        
        # Load dataset if not already loaded
        if cls._noise_dataset is None:
            # Get the dataset and the specific split
            dataset = datasets.load_dataset(**noise_audio_repo)
            # If split was specified in noise_audio_repo, it's already the right split
            # Otherwise, get the default split ('train' if available, otherwise first split)
            if 'split' not in noise_audio_repo:
                split = 'train' if 'train' in dataset else list(dataset.keys())[0]
                cls._noise_dataset = dataset[split]
            else:
                cls._noise_dataset = dataset
            
        return cls._noise_dataset


@single_batch_entry
def normalize_audio(r, src_or_tgt):
    '''Normalize audio to zero mean and max magnitude of 1.0.'''
    x = r[src_or_tgt]
    x = x - np.mean(x)
    x = x / (np.max(np.abs(x)) + 1e-3)
    r[src_or_tgt] = x
    return r


@single_batch_entry
def augment_audio_noise(r, 
                        src_or_tgt,
                        p=1.0,
                        noise_audio_repo=None,
                        max_relative_amplitude=.9,
                        max_coverage=1.0,
                        min_coverage=0.4):
    '''Add random noise to an audio sample.
    
    Args:
        r: dictionary containing fields of a single dataset row.
        src_or_tgt: str, key such that r[src_or_tgt] contains the audio array
            to be augmented.
        p: float, probability that the augmentation is applied, in range (0,1)
        noise_audio_repo: if None (default) then use synthetic white noise.
            Otherwise, if this contains a dict of valid kwargs for 
            `datasets.load_dataset()`, e.g.
            `{path='Sunbird/urban-noise', subset='small', split='train'}`,
            then noise audio will be randomly sampled from this repository.
            Assume the dataset has an `audio` feature.
        max_relative_amplitude: max noise amplitude relative to the largest
            value in the source array x. The value chosen is uniform in the
            range (0, max_amplitude_relative).
        max_coverage: largest proportion of the audio sample that will have
            noise added. A value of 1 means that potentially the entire sample
            can have noise added.
        min_coverage: smallest proportion of the audio sample that can have
            noise added. The coverage for a particular sample is randomly chosen
            in the range (min_coverage, max_coverage), and a segment of this
            length is randomly selected from the sample.
    '''
    x = r[src_or_tgt]
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    # Do nothing for empty inputs
    if not len(x):
        return r

    # Do nothing for a random proportion (1-p) of the inputs
    if np.random.random() > p:
        return r

    x_reference_amplitude = np.percentile(np.abs(x), 99)
    amplitude = np.random.uniform(0, max_relative_amplitude) * x_reference_amplitude
    coverage = np.random.uniform(min_coverage, max_coverage)
    num_samples_to_affect = int(len(x) * coverage)
    start_index = np.random.randint(0, len(x) - num_samples_to_affect)
    
    if noise_audio_repo is None:
        # Use synthetic white noise
        noise = np.random.uniform(-amplitude, amplitude, size=num_samples_to_affect)
    else:
        # Get the singleton instance and load dataset if needed
        noise_augmenter = NoiseAugmenter()
        noise_dataset = noise_augmenter.get_noise_dataset(noise_audio_repo)
        
        # Randomly select a noise sample
        noise_idx = np.random.randint(0, noise_dataset.num_rows)
        noise_sample = np.array(noise_dataset[noise_idx]['audio']['array'])
        
        # If noise sample is too short, repeat it
        if len(noise_sample) < num_samples_to_affect:
            repeats = int(np.ceil(num_samples_to_affect / len(noise_sample)))
            noise_sample = np.tile(noise_sample, repeats)
            
        # If noise sample is too long, take a random segment
        if len(noise_sample) > num_samples_to_affect:
            noise_start = np.random.randint(0, len(noise_sample) - num_samples_to_affect)
            noise_sample = noise_sample[noise_start:noise_start + num_samples_to_affect]
            
        # Normalize noise amplitude
        noise_max = np.amax(np.abs(noise_sample))
        if noise_max > 0:  # Avoid division by zero
            noise = (noise_sample / noise_max) * amplitude
        else:
            noise = np.zeros(num_samples_to_affect)

    # Apply noise to the chosen segment
    x_with_noise = np.copy(x)  # Make a copy of x to prevent altering the original
    x_with_noise[start_index:start_index + num_samples_to_affect] += noise
    
    r[src_or_tgt] = x_with_noise
    return r

@single_batch_entry
def augment_audio_time_masking(r,
                               src_or_tgt,
                               num_masks_min=2,
                               num_masks_max=4,
                               max_mask_duration_ms=100):
    """Apply time masking to an audio signal, with cutouts of random duration.

    Args:
        r: dictionary containing fields of a single dataset row.
        src_or_tgt: str, key such that r[src_or_tgt] contains the audio array
            to be augmented.
        num_masks_min, num_masks_max: int, the range of masked periods is
            randomly chosen in this range.
        max_mask_duration_ms: int, the maximum duration of a mask in
            milliseconds.
    """
    audio_masked = np.copy(r[src_or_tgt]) # Avoid modifying the original
    
    # Convert maximum mask duration from milliseconds to number of samples
    sample_rate = r[f'{src_or_tgt}.sample_rate']
    max_mask_duration_samples = int((sample_rate / 1000) * max_mask_duration_ms)
    
    num_masks = np.random.randint(num_masks_min, num_masks_max)
    total_time_steps = len(audio_masked)
    
    for _ in range(num_masks):
        mask_duration = np.random.randint(0, max_mask_duration_samples + 1)  
        if total_time_steps > mask_duration + 1:
            t0 = np.random.randint(0, total_time_steps - mask_duration + 1)
            audio_masked[t0:t0 + mask_duration] = 0

    r[src_or_tgt] = audio_masked
    return r

@single_batch_entry
def clean_and_remove_punctuation(
    r, src_or_tgt, allowed_punctuation=None, **clean_text_args):
    r[src_or_tgt] = cleantext.clean(
        r[src_or_tgt], to_ascii=False, no_punct=False, **clean_text_args)
    
    punct = list(string.punctuation)
    if allowed_punctuation:
        for allowed in allowed_punctuation:
            punct.remove(allowed)
        
    r[src_or_tgt] = ''.join([c for c in r[src_or_tgt] if c not in punct])
    return r
    
@single_batch_entry
def lower_case(r, src_or_tgt):
    r[src_or_tgt] = r[src_or_tgt].lower()
    return r

@single_batch_entry
def set_sample_rate(r, src_or_tgt, rate, p=1.0):
    '''Resamples audio data, if the sample rate in the record is different.'''
    current_sample_rate = r[f'{src_or_tgt}.sample_rate']
    if current_sample_rate != rate:
        if p == 1.0 or np.random.random() < p:
            audio_data = np.array(r[src_or_tgt])
            resampled_audio_data = librosa.resample(
                audio_data, orig_sr=current_sample_rate, target_sr=rate)
            r[src_or_tgt] = resampled_audio_data
            r[f'{src_or_tgt}.sample_rate'] = rate
    
    return r

# TODO: Check that the order of preprocessing operations makes sense. For
# example, don't call match_target_sentence_format_to_source after 
# prefix_dataset_tag (because then the tag is part of the text)
    