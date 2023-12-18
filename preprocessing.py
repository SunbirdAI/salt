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

import string
import random
import sacremoses
import functools
from utils import single_batch_entry
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

normalizer = sacremoses.MosesPunctNormalizer()


@single_batch_entry
def prefix_target_language(r, src_or_tgt):
    r[src_or_tgt] = f'>>{r["target.language"]}<< ' + r[src_or_tgt]
    return r

@single_batch_entry
def sentence_format(r, src_or_tgt, add_full_stop=True):
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
    print(r)
    char_augmenter = nac.RandomCharAug(**char_augmentation_params)
    r[src_or_tgt] = char_augmenter.augment(r[src_or_tgt])
    return r

def augment_words(r, src_or_tgt, **word_augmentation_params):
    word_augmenter = naw.RandomWordAug(**word_augmentation_params)
    r[src_or_tgt] = word_augmenter.augment(r[src_or_tgt])
    return r
