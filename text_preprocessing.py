'''
Text examples have the fields:
text, language, origin_dataset
'''
import string
import random
import sacremoses

normalizer = sacremoses.MosesPunctNormalizer()  

def prefix_target_language(r, src_or_tgt):
    for i in range(len(r['source'])):
        r[src_or_tgt][i] = f'>>{r["target_language"][i]}<< ' + r[src_or_tgt][i]
    return r

def sentence_format(r, src_or_tgt, add_full_stop=True):
    for i in range(len(r['source'])):
        text = r[src_or_tgt][i] 
        text = text[0].capitalize() + text[1:]
        if text[-1] not in ['.', '!', '?'] and add_full_stop:
            text = text + '.'
        r[src_or_tgt][i] = text
    return r

def normalise_text(r, src_or_tgt):
    for i in range(len(r['source'])):
        text = r[src_or_tgt][i]
        text = normalizer.normalize(text)
        r[src_or_tgt][i] = text
    return r    
