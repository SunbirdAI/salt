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
import string
import random
import cleantext
import functools
import numpy as np
import librosa
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

from .utils import single_batch_entry



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
def clean_text(r, src_or_tgt, **clean_text_args):
    r[src_or_tgt] = cleantext.clean(
        r[src_or_tgt], to_ascii=False, lower=False, **clean_text_args)
    return r

def augment_characters(r, src_or_tgt, **char_augmentation_params):
    char_augmenter = nac.RandomCharAug(**char_augmentation_params)
    r[src_or_tgt] = char_augmenter.augment(r[src_or_tgt])
    return r

def augment_words(r, src_or_tgt, **word_augmentation_params):
    word_augmenter = naw.RandomWordAug(**word_augmentation_params)
    r[src_or_tgt] = word_augmenter.augment(r[src_or_tgt])
    return r

@single_batch_entry
def clean_and_remove_punctuation(r, src_or_tgt, **clean_text_args):
    r[src_or_tgt] = cleantext.clean(
        r[src_or_tgt], to_ascii=False, no_punct=True, **clean_text_args)
    # The cleantext library doesn't remove all punctuation marks.
    r[src_or_tgt] = r[src_or_tgt].translate(
        str.maketrans('', '', string.punctuation))
    return r
    
@single_batch_entry
def lower_case(r, src_or_tgt):
    r[src_or_tgt] = r[src_or_tgt].lower()
    return r

@single_batch_entry
def random_capitalise_source_and_target(r, src_or_tgt, p=0.005):
    if np.random.random() < p:
        r['source'] = r['source'].upper()
        r['target'] = r['target'].upper()
    return r

@single_batch_entry
def set_sample_rate(r, src_or_tgt, rate):
    '''Resamples audio data, if the sample rate in the record is different.'''
    current_sample_rate = r[f'{src_or_tgt}.sample_rate']
    if current_sample_rate != rate:
        audio_data = np.array(r[src_or_tgt])
        resampled_audio_data = librosa.resample(
            audio_data, orig_sr=current_sample_rate, target_sr=rate)
        r[src_or_tgt] = resampled_audio_data
        r[f'{src_or_tgt}.sample_rate'] = rate
    
    return r

# TODO: Check that the order of preprocessing operations makes sense. For
# example, don't call match_target_sentence_format_to_source after 
# prefix_dataset_tag (because then the tag is part of the text)
    