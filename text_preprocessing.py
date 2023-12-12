'''
Text examples have the fields:
text, language, origin_dataset
'''
import string
import random
import sacremoses
import functools
from utils import single_batch_entry

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
