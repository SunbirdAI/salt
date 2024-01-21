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
  'target': {'array': [...], sampling_rate=16000},
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
import sacremoses
import functools
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

from .utils import single_batch_entry

normalizer = sacremoses.MosesPunctNormalizer()


@single_batch_entry
def prefix_target_language(r, src_or_tgt):
    r[src_or_tgt] = f'>>{r["target.language"]}<< ' + r[src_or_tgt]
    return r

@single_batch_entry
def sentence_format(r, src_or_tgt, add_full_stop=True):
    '''Begin with a capital letter, and optionally end with full stop.'''
    text = r[src_or_tgt]
    text = text[0].capitalize() + text[1:]
    if text[-1] not in ['.', '!', '?'] and add_full_stop:
        text = text + '.'
    r[src_or_tgt] = text
    return r

@single_batch_entry
def normalise_text(r, src_or_tgt):
    r[src_or_tgt] = normalizer.normalize(r[src_or_tgt])
    return r    

def augment_characters(r, src_or_tgt, **char_augmentation_params):
    char_augmenter = nac.RandomCharAug(**char_augmentation_params)
    r[src_or_tgt] = char_augmenter.augment(r[src_or_tgt])
    return r

def augment_words(r, src_or_tgt, **word_augmentation_params):
    word_augmenter = naw.RandomWordAug(**word_augmentation_params)
    r[src_or_tgt] = word_augmenter.augment(r[src_or_tgt])
    return r

def remove_punctuation(r, src_or_tgt, punctuation_chars=None):
    default_punctuation_chars = [
        ",",
        "?",
        ".",
        "!",
        "-",
        ";",
        ":",
        '""',
        "%",
        "'",
        '"',
        "�",
        "'",
        "\u2018",
        "\u2019",
    ]
    chars_to_remove = punctuation_chars or default_punctuation_chars
    chars_to_remove_regex = f'[{"".join(chars_to_remove)}]'
    
    for i in range(len(r[src_or_tgt])):
        r[src_or_tgt][i] = re.sub(chars_to_remove_regex, "", r[src_or_tgt][i])
        
    return r
    
@single_batch_entry
def lower_case(r, src_or_tgt):
    r[src_or_tgt] = r[src_or_tgt].lower()
    return r


    