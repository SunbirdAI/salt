"""Fine-tune NLLB-200 3.3B on the 69 salt-69 languages, and evaluate on salt-69.

Successor to translation_nllb_training_with_ug40_languages.ipynb, using the
consolidated Sunbird datasets:

- Sunbird/sunflower-pretrain-data, config parallel_text: multi-way parallel
  rows as a JSON dict {lang: text}. Large but less filtered. Sampled with a cap
  per eng->xx / xx->eng direction plus some non-English pairs. Keys ending in
  "_target" (e.g. kin_target) are translations of English where gender
  information was lost ("he went"/"she went" -> one form), so they're only used
  as eng->xx targets.
- Sunbird/sunflower-posttrain-data, config sft_translations: refined chat-format
  pairs. The source text is extracted from the prompt, and direction is
  languages[0] -> languages[1] only. The test split is salt-69, so it's unused.
  More heavily quality-filtered than parallel_text, and in modern usage: for
  some languages parallel_text has only archaic forms (old bible translations),
  which are useful early in training but shouldn't shape the final output.

Training runs in one phase by default. With --sft_epoch it runs two: phase 1
on all data, then phase 2 on the sft pairs alone at a lower learning rate, so
the clean, modern data is what the last optimizer steps see. Phase 2 reuses
the sft rows of the same prepared dataset, so `prepare` is unchanged. Phase 1
keeps its best checkpoint by dev metric; phase 2 takes its last one, because
its training data is the same distribution as the dev set, so the dev metric
rises throughout and "best" would only mean "last evaluated". Note that the
sft data doesn't cover every language (nothing for kln, ber, ndp, dga; no
xx->eng for led, ikx, din; almost no non-English directions), so compare the
per-language salt-69 table against phase1/ to see what phase 2 cost.

Evaluation computes chrF/BLEU on Sunbird/salt-69 for every eng->xx and xx->eng
direction. The metric matches sunflower/eval_translation.py (whisper
BasicTextNormalizer, sentence-level sacrebleu scores averaged per direction),
and the results JSON uses the same schema, so scores compare directly with
LLM results from /sunflower/scripts/eval/translation.py.

During training, each evaluation (at step 0 unless --no-eval_on_start, then
every --eval_steps) reports eval_loss and
greedy-decoded chrF on the sft_translations dev split: eval_chrf over all
directions, plus eval_chrf_to_eng / _from_eng / _non_eng. By default the best
checkpoint and early stopping use eval_chrf (--best_metric). salt-69 is only
used in the final evaluation, so it never influences model selection.

Language tokens come from constants.SALT_LANGUAGE_CODES_NLLB_69: NLLB's own
code where it supports the language, otherwise a repurposed code. The final
model directory includes nllb_language_codes.json with this mapping.

Optional source language: with --src_lang_dropout (default 0.25), that
fraction of training pairs get the unknown-source token "und"
(constants.NLLB_69_UNKNOWN_SOURCE_CODE) instead of their source-language
token, so at inference the source language can be given or left out (e.g. for
code-switched text, or when it isn't known). The target language is always
required. Dev chrF is then reported both ways (eval_chrf and eval_chrf_nosrc),
and the salt-69 eval runs both ways (--source_modes): NAME_results.json with
the source given, NAME_nosrc_results.json without, plus a per-language
comparison table. translate(..., src=None, ...) translates without a source
language.

Long inputs: NLLB's tokenizer turns newlines into spaces, and the model was
trained on <=512-token inputs. Multi-line SFT dialogues are trained both whole
and split by line, so the model handles a whole transcript. To keep line
breaks in production output, split on newlines, translate the lines as one
batch, and re-join.

Punctuation: all training text and model inputs go through normalize_text
(NLLB's Moses punctuation normalization), because NLLB's vocabulary has no
typographic quotes or dashes. Output therefore uses ASCII punctuation
(' " -). Production inference should apply normalize_text to its inputs too.

Requires torch, transformers, datasets, sacrebleu, sacremoses and torchao.

Usage (from the repo root, in /venv/main):
  python notebooks/training/translation_nllb_training_69_languages.py all \
      --output_dir /workspace/nllb-3.3b-salt69
  # one epoch on all data, then one on the sft data alone:
  python notebooks/training/translation_nllb_training_69_languages.py all \
      --output_dir /workspace/nllb-3.3b-salt69-sft \
      --sft_upsample 1 --num_train_epochs 1 --sft_epoch
  # just the sft phase, on top of a model already trained on all data:
  python notebooks/training/translation_nllb_training_69_languages.py all \
      --output_dir /workspace/nllb-1.3b-sft --skip_phase1 \
      --base_model Sunbird-experimental/nllb1.3b-67languages --sft_upsample 1
  # or step by step: prepare, train, eval (each takes --output_dir)
  python notebooks/training/translation_nllb_training_69_languages.py eval \
      --output_dir /workspace/eval-base --model facebook/nllb-200-3.3B

MLflow logging is enabled when MLFLOW_TRACKING_URI is set.
"""

import argparse
import collections
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import datasets
import numpy as np
import pandas as pd
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import constants  # noqa: E402

log = logging.getLogger('nllb69')

BASE_MODEL = 'facebook/nllb-200-3.3B'
PARALLEL_DATA = ('Sunbird/sunflower-pretrain-data', 'parallel_text')
SFT_DATA = ('Sunbird/sunflower-posttrain-data', 'sft_translations')
EVAL_DATA = ('Sunbird/salt-69', 'sunflower', 'test')
LANGS = constants.SALT_LANGUAGE_CODES_NLLB_69
UNKNOWN_SOURCE = constants.NLLB_69_UNKNOWN_SOURCE  # lang_ids key, source only
LANGUAGE_NAMES = constants.SALT_LANGUAGE_NAMES


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

# Prompt templates in sft_translations; group 1 is the source text.
SFT_PROMPT_PATTERNS = [re.compile(p, re.S) for p in (
    # "Translate to X: ..." / "Translate from X to Y: ..."
    r'^Translate (?:from [^:\n]+? )?to [^:\n]+?:\s*(.*)$',
    # "Translate this X sentence into Y. Give only the translation.\n\n..."
    r'^Translate this [^\n]+? into [^\n]+?\. Give only the translation\.\s*(.*)$',
    # "What is the Y translation of this X sentence?\n\n..."
    r'^What is the [^\n]+? translation of this [^\n]+? sentence\?\s*(.*)$',
    # "How would the X text: ... be translated into Y?"
    r'^How would the [^:\n]+? text: (.*) be translated into [^?\n]+\?\s*$',
    # "[You are a translation expert. ...\n]X sentence: ...\nY sentence: . Return only ..."
    r'^(?:You are a translation expert\.[^\n]*\n)?[^\n:]+ sentence: (.*?)\s*'
    r'\n[^\n:]+ sentence: \. Return only the translated sentence\.\s*$',
)]


def extract_sft_source(prompt):
    for pattern in SFT_PROMPT_PATTERNS:
        match = pattern.match(prompt)
        if match:
            return match.group(1).strip()
    return None


def nonempty_lines(text):
    return [line.strip() for line in text.split('\n') if line.strip()]


_punct_normalizer = None


def normalize_text(text):
    '''NLLB-style text normalization, applied to all training text and to
    model inputs.

    NLLB's vocabulary was built from text normalized with Moses
    normalize-punctuation, so typographic punctuation is missing from it:
    ’ ‘ “ ” – — « » become <unk>. Left as is, the model learns to output
    <unk>, and decoding silently deletes it ("difference—it's" ->
    "differenceit's"). This maps them to the ASCII forms NLLB knows
    (’ -> ', “ ” « » -> ", — -> " - ", – -> -, … -> ...) and removes control
    characters. Moses' quote/comma reordering and number rules are turned off
    because they're language-specific.

    Also removes two combining marks NLLB can't represent: U+0307 (stray dot,
    mostly Lugungu "i̇") and U+0326 (Kabyle text artifacts).'''
    global _punct_normalizer
    if _punct_normalizer is None:
        from sacremoses import MosesPunctNormalizer
        _punct_normalizer = MosesPunctNormalizer(
            pre_replace_unicode_punct=True, post_remove_control_chars=True,
            norm_quote_commas=False, norm_numbers=False)
    # Whitespace first: remove_control_chars would delete \n and \t outright.
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('̇', '').replace('̦', '')
    return _punct_normalizer.normalize(text)


def salt69_texts():
    '''All salt-69 texts, normalized as training text is, for the leakage
    filter.'''
    ds = datasets.load_dataset(EVAL_DATA[0], EVAL_DATA[1], split=EVAL_DATA[2])
    return {normalize_text(t) for col in ds.column_names for t in ds[col] if t}


class Pairs:
    '''Accumulates translation pairs as columns.'''
    COLUMNS = ['source', 'target', 'source.language', 'target.language',
               'origin']

    def __init__(self):
        self.data = {c: [] for c in self.COLUMNS}

    def add(self, src_lang, tgt_lang, source, target, origin):
        for col, value in zip(self.COLUMNS,
                              (source, target, src_lang, tgt_lang, origin)):
            self.data[col].append(value)

    def __len__(self):
        return len(self.data['source'])


def add_parallel_pairs(pairs, args, rng):
    ds = datasets.load_dataset(*PARALLEL_DATA, split='train')
    texts = ds['text']
    if args.limit_rows:
        texts = texts[:args.limit_rows]
    log.info('Parsing %d parallel_text rows', len(texts))

    # rows[i] = (clean {lang: text}, target_only {lang: text})
    rows = []
    eng_to = collections.defaultdict(list)  # lang -> [row*2 + is_target_only]
    to_eng = collections.defaultdict(list)  # lang -> [row]
    multi = []  # rows with at least two clean non-English languages
    for i, text in enumerate(texts):
        clean, target_only = {}, {}
        for key, value in json.loads(text).items():
            if not isinstance(value, str) or not value.strip():
                continue
            if key.endswith('_target'):
                if key[:-len('_target')] in LANGS:
                    target_only[key[:-len('_target')]] = value.strip()
            elif key in LANGS:
                clean[key] = value.strip()
        rows.append((clean, target_only))
        if 'eng' in clean:
            for lang in clean:
                if lang != 'eng':
                    eng_to[lang].append(i * 2)
                    to_eng[lang].append(i)
            for lang in target_only:
                eng_to[lang].append(i * 2 + 1)
        if len(clean) - ('eng' in clean) >= 2:
            multi.append(i)

    cap = args.max_per_direction
    for lang in sorted(eng_to):
        pool = eng_to[lang]
        for code in rng.sample(pool, min(cap, len(pool))):
            clean, target_only = rows[code // 2]
            target = target_only[lang] if code % 2 else clean[lang]
            pairs.add('eng', lang, clean['eng'], target, 'parallel')
    for lang in sorted(to_eng):
        pool = to_eng[lang]
        for i in rng.sample(pool, min(cap, len(pool))):
            pairs.add(lang, 'eng', rows[i][0][lang], rows[i][0]['eng'],
                      'parallel')

    n_non_eng = int(args.non_eng_fraction * len(pairs))
    rng.shuffle(multi)
    pair_counts = collections.Counter()
    added = 0
    for i in multi:
        if added >= n_non_eng:
            break
        clean = rows[i][0]
        a, b = rng.sample(sorted(l for l in clean if l != 'eng'), 2)
        if pair_counts[(a, b)] >= args.max_per_non_eng_pair:
            continue
        pair_counts[(a, b)] += 1
        pairs.add(a, b, clean[a], clean[b], 'parallel_x')
        added += 1
    log.info('parallel_text: %d English-pivot + %d non-English pairs '
             '(%d distinct non-English directions)',
             len(pairs) - added, added, len(pair_counts))


def add_sft_pairs(pairs, split, upsample):
    ds = datasets.load_dataset(*SFT_DATA, split=split)
    skipped = collections.Counter()
    for messages, languages in zip(ds['messages'], ds['languages']):
        src_lang, tgt_lang = languages
        if src_lang not in LANGS or tgt_lang not in LANGS:
            skipped['unknown language'] += 1
            continue
        prompt = next(m['content'] for m in messages if m['role'] == 'user')
        target = next(m['content'] for m in messages
                      if m['role'] == 'assistant')
        source = extract_sft_source(prompt)
        if not source:
            skipped['unparsed prompt'] += 1
            continue
        source_lines, target_lines = nonempty_lines(source), nonempty_lines(target)
        # Whole example (newlines become spaces, as NLLB's tokenizer does).
        for _ in range(upsample):
            pairs.add(src_lang, tgt_lang, ' '.join(source_lines),
                      ' '.join(target_lines), 'sft')
        # Line-aligned dialogues also as one pair per line.
        if len(source_lines) > 1 and len(source_lines) == len(target_lines):
            for s, t in zip(source_lines, target_lines):
                pairs.add(src_lang, tgt_lang, s, t, 'sft_line')
    log.info('sft_translations/%s: %d pairs so far; skipped %s',
             split, len(pairs), dict(skipped))


def filter_pairs(pairs, eval_texts, max_ratio=3.0):
    '''Normalizes text (normalize_text), then drops empty, identical,
    eval-leaked, duplicate (parallel only) and length-mismatched (parallel
    only) pairs.'''
    keep = Pairs()
    seen = set()
    dropped = collections.Counter()
    d = pairs.data
    for src, tgt, sl, tl, origin in zip(*(d[c] for c in Pairs.COLUMNS)):
        src, tgt = normalize_text(src), normalize_text(tgt)
        if not src or not tgt or src == tgt:
            dropped['empty/identical'] += 1
            continue
        if src in eval_texts or tgt in eval_texts:
            dropped['in salt-69'] += 1
            continue
        if origin.startswith('parallel'):
            key = (sl, tl, src, tgt)
            if key in seen:
                dropped['duplicate'] += 1
                continue
            seen.add(key)
            shorter, longer = sorted((len(src), len(tgt)))
            if shorter > 20 and longer / shorter > max_ratio:
                dropped['length ratio'] += 1
                continue
        keep.add(sl, tl, src, tgt, origin)
    log.info('Filtered: kept %d, dropped %s', len(keep), dict(dropped))
    return keep


def tokenize_batch(batch, tokenizer, lang_ids):
    enc = tokenizer(batch['source'], text_target=batch['target'])
    for i in range(len(enc['input_ids'])):
        # NLLB puts the language token first; set the real languages.
        enc['input_ids'][i][0] = lang_ids[batch['source.language'][i]]
        enc['labels'][i][0] = lang_ids[batch['target.language'][i]]
    enc['n_source_tokens'] = [len(x) for x in enc['input_ids']]
    enc['n_target_tokens'] = [len(x) for x in enc['labels']]
    return enc


def tokenize_and_length_filter(pairs, tokenizer, lang_ids, args, name):
    ds = datasets.Dataset.from_dict(pairs.data)
    # Single process: the fast tokenizer is already multi-threaded, and
    # num_proc>1 was ~10x slower here (per-worker tokenizer overhead).
    ds = ds.map(tokenize_batch, batched=True,
                fn_kwargs={'tokenizer': tokenizer, 'lang_ids': lang_ids},
                desc=f'Tokenizing {name}')
    df = ds.select_columns(['source.language', 'target.language',
                            'n_source_tokens', 'n_target_tokens']).to_pandas()
    n_src, n_tgt = df['n_source_tokens'].values, df['n_target_tokens'].values
    too_long = (n_src > args.max_length) | (n_tgt > args.max_length)
    if too_long.any():
        df['long'] = too_long
        by_dir = df.groupby(['source.language', 'target.language'])[
            'long'].agg(['sum', 'mean'])
        by_dir = by_dir[by_dir['sum'] > 0].sort_values('mean', ascending=False)
        log.info('%s: dropping %d pairs over %d tokens. Directions most '
                 'affected (count, fraction):\n%s', name, too_long.sum(),
                 args.max_length, by_dir.head(15).to_string())
    ds = ds.select(np.flatnonzero(~too_long))
    for label, n in (('source', n_src[~too_long]), ('target', n_tgt[~too_long])):
        log.info('%s %s tokens: p50=%d p90=%d p99=%d max=%d', name, label,
                 *np.percentile(n, [50, 90, 99]), n.max())
    return ds


def write_pair_counts(ds, path):
    df = ds.select_columns(
        ['source.language', 'target.language', 'origin']).to_pandas()
    counts = (df.value_counts().unstack('origin', fill_value=0)
              .assign(total=lambda x: x.sum(axis=1))
              .sort_values('total', ascending=False))
    counts.to_csv(path)
    log.info('Pairs by origin: %s', df['origin'].value_counts().to_dict())
    eng_to = counts['total'].xs('eng', level='source.language')
    to_eng = counts['total'].xs('eng', level='target.language')
    summary = pd.DataFrame({'eng->xx': eng_to, 'xx->eng': to_eng}).fillna(0)
    summary = summary.astype(int).sort_values('eng->xx')
    log.info('Pairs per language (all origins):\n%s', summary.to_string())
    missing = sorted(set(LANGS) - {'eng'} - set(summary.index))
    if missing:
        log.warning('No English-pivot training data for: %s', missing)


def prepare(args):
    rng = random.Random(args.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    lang_ids = constants.nllb_69_language_token_ids(tokenizer)
    eval_texts = salt69_texts()

    pairs = Pairs()
    add_parallel_pairs(pairs, args, rng)
    add_sft_pairs(pairs, 'train', upsample=args.sft_upsample)
    train = tokenize_and_length_filter(filter_pairs(pairs, eval_texts),
                                       tokenizer, lang_ids, args, 'train')

    val_pairs = Pairs()
    add_sft_pairs(val_pairs, 'dev', upsample=1)
    validation = tokenize_and_length_filter(
        filter_pairs(val_pairs, eval_texts), tokenizer, lang_ids, args,
        'validation')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets.DatasetDict({'train': train, 'validation': validation}
                         ).save_to_disk(str(out / 'pairs'))
    write_pair_counts(train, out / 'pair_counts.csv')
    log.info('Saved %d train / %d validation pairs to %s',
             len(train), len(validation), out / 'pairs')
    for _, row in train.shuffle(seed=args.seed).select(range(min(10, len(train)))).to_pandas().iterrows():
        log.info('  [%s] %s -> %s | %s => %s', row['origin'],
                 row['source.language'], row['target.language'],
                 row['source'][:100], row['target'][:100])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def mlflow_enabled():
    return bool(os.environ.get('MLFLOW_TRACKING_URI'))


class Seq2SeqCollator:
    '''Pads the batch and adds decoder_input_ids (labels shifted right).

    DataCollatorForSeq2Seq only does the shift if the model has
    prepare_decoder_input_ids_from_labels, which M2M100 lacks in transformers
    5.x. The model would shift the labels itself, but token_level_loss means
    the Trainer doesn't pass it the labels.

    src_lang_dropout: the fraction of pairs whose source-language token is
    replaced by unknown_source_id, so the model learns to translate without
    being told the source language. Use it for training batches only, so eval
    loss is deterministic.'''

    def __init__(self, tokenizer, config, src_lang_dropout=0.0,
                 unknown_source_id=None):
        self.pad = transformers.DataCollatorForSeq2Seq(
            tokenizer, label_pad_token_id=-100)
        self.pad_token_id = config.pad_token_id
        self.decoder_start_token_id = config.decoder_start_token_id
        self.src_lang_dropout = src_lang_dropout
        self.unknown_source_id = unknown_source_id
        if src_lang_dropout:
            assert unknown_source_id is not None

    def __call__(self, features):
        from transformers.models.m2m_100.modeling_m2m_100 import (
            shift_tokens_right)
        batch = self.pad(features)
        if self.src_lang_dropout:
            # The source-language token is always at position 0.
            drop = torch.rand(len(batch['input_ids'])) < self.src_lang_dropout
            batch['input_ids'][drop, 0] = self.unknown_source_id
        batch['decoder_input_ids'] = shift_tokens_right(
            batch['labels'], self.pad_token_id, self.decoder_start_token_id)
        return batch


def token_level_loss(outputs, labels, num_items_in_batch=None):
    '''Cross-entropy summed over tokens, divided by the target-token count of
    the whole accumulated batch.

    Needed because M2M100's forward() accepts **kwargs, so the Trainer assumes
    its loss is already normalized for gradient accumulation and skips the
    division, but M2M100 returns a per-micro-batch mean. Without this, the loss
    is logged (and the gradient summed) gradient_accumulation_steps times too
    large. It also weights every target token equally across micro-batches.'''
    logits = outputs.logits
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1),
        ignore_index=-100, reduction='sum')
    if num_items_in_batch is None:  # evaluation
        num_items_in_batch = (labels != -100).sum()
    return loss / num_items_in_batch


def dev_chrf(model, tokenizer, lang_ids, dev, batch_size, max_length,
             unknown_source=False):
    '''Greedy-decodes the dev pairs and returns chrF (same sentence-level
    metric as the salt-69 eval) averaged per direction, then over all
    directions and over the to-English / from-English / non-English groups.
    The dev set has only ~5 pairs per direction, so read the averages, not
    single directions. unknown_source: translate without giving the source
    language.'''
    preds = translate(model, tokenizer, lang_ids, dev['source'],
                      None if unknown_source else dev['source.language'],
                      dev['target.language'],
                      batch_size=batch_size, num_beams=1,
                      max_new_tokens=max_length, max_length=max_length,
                      length_cap=3.0)
    by_direction = collections.defaultdict(list)
    for pred, ref, src, tgt in zip(preds, dev['target'], dev['source.language'],
                                   dev['target.language']):
        by_direction[(src, tgt)].append(
            compute_sentence_metrics(pred, ref)['chrf'])
    means = {d: float(np.mean(v)) for d, v in by_direction.items()}
    groups = {
        'chrf': list(means.values()),
        'chrf_to_eng': [v for (s, t), v in means.items() if t == 'eng'],
        'chrf_from_eng': [v for (s, t), v in means.items() if s == 'eng'],
        'chrf_non_eng': [v for (s, t), v in means.items()
                         if 'eng' not in (s, t)],
    }
    return {k: round(float(np.mean(v)), 4) for k, v in groups.items() if v}


class TokenBudgetBatchSampler(torch.utils.data.Sampler):
    '''Batches of similar-length pairs, sized so that
    batch size x (longest source + longest target) <= max_tokens.

    Short sentences go many to a batch and long dialogues few, so each batch
    does a similar amount of GPU work and peak memory stays bounded. With
    fixed-size batches, most batches were tiny (8 x ~65 tokens) and the GPU sat
    idle waiting for kernel launches. Pairs are sorted by length within
    random megabatches, batch order is shuffled per epoch, and the batch with
    the longest sequence goes first, so an out-of-memory error happens at
    once rather than hours in.'''

    def __init__(self, source_lengths, target_lengths, max_tokens,
                 max_batch_size, seed, megabatch_size=100_000):
        self.source_lengths = np.asarray(source_lengths)
        self.target_lengths = np.asarray(target_lengths)
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.seed = seed
        self.megabatch_size = megabatch_size
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.batches = self._make_batches()

    def _make_batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        src, tgt = self.source_lengths, self.target_lengths
        order = rng.permutation(len(src))
        batches = []
        for start in range(0, len(order), self.megabatch_size):
            mega = order[start:start + self.megabatch_size]
            mega = mega[np.argsort(-(src[mega] + tgt[mega]), kind='stable')]
            batch, max_src, max_tgt = [], 0, 0
            for i in mega:
                s, t = max(max_src, src[i]), max(max_tgt, tgt[i])
                if batch and ((len(batch) + 1) * (s + t) > self.max_tokens
                              or len(batch) >= self.max_batch_size):
                    batches.append(batch)
                    batch, s, t = [], src[i], tgt[i]
                batch.append(int(i))
                max_src, max_tgt = s, t
            if batch:
                batches.append(batch)
        batches = [batches[j] for j in rng.permutation(len(batches))]
        # Put the most memory-hungry batches first: the most padded target
        # tokens (the batch x length x 256k-vocab logits usually dominate),
        # then the longest sequence (attention and activations).
        target_tokens = [len(b) * tgt[b].max() for b in batches]
        longest = [(src[b] + tgt[b]).max() for b in batches]
        first = [int(np.argmax(target_tokens)), int(np.argmax(longest))]
        first = list(dict.fromkeys(first))
        rest = [b for j, b in enumerate(batches) if j not in first]
        return [batches[j] for j in first] + rest

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class ThroughputCallback(transformers.TrainerCallback):
    '''Logs non-padding source tokens per second between log steps.'''

    def __init__(self):
        self.last = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or 'loss' not in logs:
            return
        now, seen = time.time(), state.num_input_tokens_seen
        if self.last is not None and now > self.last[0]:
            log.info('step %d: %.0f source tokens/s', state.global_step,
                     (seen - self.last[1]) / (now - self.last[0]))
        self.last = (now, seen)


class ChrfTrainer(transformers.Trainer):
    '''Trainer whose evaluation also reports greedy-decoded dev-set chrF
    (eval_chrf, ...). Added inside evaluation_loop so the metrics are there
    when the Trainer logs them, checks early stopping and picks the best
    checkpoint.

    With dev_chrf_unknown_source, it also reports eval_chrf_nosrc (and
    _nosrc_to_eng etc.): the same dev pairs translated without the source
    language. train_collator (e.g. with source-language dropout) is used for
    training batches; data_collator for eval.'''

    def __init__(self, *args, dev_pairs, lang_ids, max_length,
                 dev_chrf_batch_size, train_batch_sampler=None,
                 train_collator=None, dev_chrf_unknown_source=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_pairs = dev_pairs
        self.lang_ids = lang_ids
        self.max_length = max_length
        self.dev_chrf_batch_size = dev_chrf_batch_size
        self.train_batch_sampler = train_batch_sampler
        self.train_collator = train_collator or self.data_collator
        self.dev_chrf_unknown_source = dev_chrf_unknown_source

    def get_train_dataloader(self):
        if self.train_batch_sampler is None:
            # The DataLoader keeps the collator it's built with.
            eval_collator, self.data_collator = (self.data_collator,
                                                 self.train_collator)
            try:
                return super().get_train_dataloader()
            finally:
                self.data_collator = eval_collator
        dataset = self._remove_unused_columns(self.train_dataset,
                                              description='training')
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=self.train_batch_sampler,
            collate_fn=self.train_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers)
        # With one GPU, accelerate passes variable-size batches through
        # unchanged (the sampler has no batch_size attribute).
        return self.accelerator.prepare(loader)

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        prefix = kwargs.get('metric_key_prefix', 'eval')
        if prefix == 'eval':
            start = time.time()
            self.model.eval()
            for unknown_source in ([False, True] if self.dev_chrf_unknown_source
                                   else [False]):
                scores = dev_chrf(self.model, self.processing_class,
                                  self.lang_ids, self.dev_pairs,
                                  batch_size=self.dev_chrf_batch_size,
                                  max_length=self.max_length,
                                  unknown_source=unknown_source)
                if unknown_source:
                    scores = {k.replace('chrf', 'chrf_nosrc'): v
                              for k, v in scores.items()}
                output.metrics.update(
                    {f'{prefix}_{k}': v for k, v in scores.items()})
            output.metrics[f'{prefix}_chrf_runtime'] = round(
                time.time() - start, 1)
        return output


def make_batch_sampler(lengths, args, name):
    '''TokenBudgetBatchSampler over a pandas frame of token counts, or None
    when token-budget batching is off.'''
    if not args.max_tokens_per_batch:
        return None
    sampler = TokenBudgetBatchSampler(
        lengths['n_source_tokens'].values,
        lengths['n_target_tokens'].values,
        max_tokens=args.max_tokens_per_batch,
        max_batch_size=args.max_sequences_per_batch, seed=args.seed)
    log.info('%s: token-budget batching, %d batches of <= %d padded tokens '
             '(%.1f pairs per batch on average)', name, len(sampler),
             args.max_tokens_per_batch, len(lengths) / len(sampler))
    return sampler


def make_trainer(args, model, tokenizer, lang_ids, train_ds, batch_sampler,
                 val_ds, dev_pairs, *, checkpoints_dir, learning_rate,
                 num_train_epochs, warmup_steps, eval_steps, select_best,
                 run_name, is_final_phase):
    '''Builds a ChrfTrainer for one training phase.

    select_best: keep the best checkpoint by dev metric and stop early.
    Phase 2 sets it False, because its training data is the same
    distribution as the dev set, so the dev metric rises monotonically and
    "best" would just mean "last evaluated". The last checkpoint is taken
    instead.'''
    import torchao.optim

    # Pure-bf16 weights: stochastic rounding stops small updates being
    # rounded away; 8-bit optimizer states keep memory within one 32GB GPU.
    optimizer = torchao.optim.AdamW8bit(
        model.parameters(), lr=learning_rate,
        weight_decay=args.weight_decay, bf16_stochastic_round=True)

    training_args = transformers.TrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type='linear',
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        max_steps=args.max_steps,
        # The weights are already bf16. bf16=True would add autocast, and
        # accelerate then copies the (batch x length x 256k vocab) logits
        # to fp32, which needs ~8GB more at 16x512 tokens.
        bf16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        # Only used without token-budget batching.
        train_sampling_strategy='group_by_length',
        include_num_input_tokens_seen='non_padding',
        eval_strategy='steps',
        eval_steps=eval_steps,
        eval_on_start=args.eval_on_start,
        save_strategy='steps',
        save_steps=eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=select_best,
        # Dev-set chrF (higher is better) or loss (lower is better); the
        # Trainer infers the direction from the name.
        metric_for_best_model=f'eval_{args.best_metric}',
        logging_steps=args.logging_steps,
        dataloader_num_workers=4,
        report_to=['mlflow'] if mlflow_enabled() else 'none',
        run_name=run_name,
        seed=args.seed,
        # Only the last phase pushes, so the hub gets one upload.
        push_to_hub=bool(args.hub_model_id) and is_final_phase,
        hub_model_id=args.hub_model_id,
        hub_strategy='end',
        hub_private_repo=True,
    )
    callbacks = [ThroughputCallback()]
    if select_best:
        callbacks.insert(0, transformers.EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience))
    return ChrfTrainer(
        dev_pairs=dev_pairs,
        lang_ids=lang_ids,
        max_length=args.max_length,
        dev_chrf_batch_size=args.dev_chrf_batch_size,
        train_batch_sampler=batch_sampler,
        train_collator=Seq2SeqCollator(
            tokenizer, model.config, src_lang_dropout=args.src_lang_dropout,
            unknown_source_id=lang_ids[UNKNOWN_SOURCE]),
        dev_chrf_unknown_source=args.src_lang_dropout > 0,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Seq2SeqCollator(tokenizer, model.config),
        processing_class=tokenizer,
        compute_loss_func=token_level_loss,
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )


def free_trainer(trainer):
    '''Drops a finished trainer and its 8-bit optimizer states (~2 bytes per
    parameter, so several GB for a 3.3B model) before the next phase builds
    its own. The model itself is kept: the next phase continues from it.'''
    import gc
    trainer.model = None
    trainer.optimizer = None
    trainer.lr_scheduler = None
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(args):
    out = Path(args.output_dir)
    pairs = datasets.load_from_disk(str(out / 'pairs'))
    model_columns = ['input_ids', 'attention_mask', 'labels']
    train_ds = pairs['train'].select_columns(model_columns)
    lengths = pairs['train'].select_columns(
        ['n_source_tokens', 'n_target_tokens']).to_pandas()
    batch_sampler = None
    if not args.skip_phase1:
        batch_sampler = make_batch_sampler(lengths, args, 'phase 1 (all data)')
    val_ds = pairs['validation'].select_columns(model_columns)
    dev_pairs = pairs['validation'].select_columns(
        ['source', 'target', 'source.language', 'target.language']).to_dict()
    log.info('Prepared %d train pairs, validating on %d', len(train_ds),
             len(val_ds))

    # Phase 2 trains on the sft rows of the same prepared dataset ('sft' and
    # 'sft_line'), so no second `prepare` is needed.
    sft_ds = sft_sampler = None
    if args.sft_epoch:
        origins = np.asarray(pairs['train']['origin'])
        sft_idx = np.flatnonzero(np.char.startswith(origins, 'sft'))
        if not len(sft_idx):
            raise SystemExit(
                '--sft_epoch: no sft rows in the prepared data. Re-run '
                '`prepare` with --sft_upsample >= 1.')
        sft_ds = train_ds.select(sft_idx)
        sft_sampler = make_batch_sampler(
            lengths.iloc[sft_idx].reset_index(drop=True), args,
            'phase 2 (sft only)')
        log.info('Phase 2 will train on %d sft pairs%s, origins: %s',
                 len(sft_idx),
                 '' if args.skip_phase1 else
                 ' (%.1f%% of phase 1)' % (100 * len(sft_idx) / len(train_ds)),
                 dict(collections.Counter(origins[sft_idx].tolist())))

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16)
    lang_ids = constants.nllb_69_language_token_ids(tokenizer)
    if args.src_lang_dropout:
        log.info('Source-language dropout: %.0f%% of training pairs get the '
                 'unknown-source token (%s)', 100 * args.src_lang_dropout,
                 constants.NLLB_69_UNKNOWN_SOURCE_CODE)

    if mlflow_enabled():
        # Start the run here (the Trainer's callback reuses an active run and
        # leaves it open) so `all` logs training and eval to one run.
        import mlflow
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=out.name)
        mlflow.log_artifact(str(out / 'pair_counts.csv'))

    trainer = None
    if not args.skip_phase1:
        log.info('Phase 1: all data, %g epoch(s) at lr %g',
                 args.num_train_epochs, args.learning_rate)
        trainer = make_trainer(
            args, model, tokenizer, lang_ids, train_ds, batch_sampler, val_ds,
            dev_pairs, checkpoints_dir=out / 'checkpoints',
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            warmup_steps=args.warmup_steps, eval_steps=args.eval_steps,
            select_best=True, run_name=out.name,
            is_final_phase=not args.sft_epoch)
        trainer.train()
        if torch.cuda.is_available():
            log.info('Phase 1 peak GPU memory: %.1f GB',
                     torch.cuda.max_memory_allocated() / 1e9)
        if args.sft_epoch:
            # Phase 1 reloaded its best checkpoint into `model`, so phase 2
            # continues from that, with a fresh optimizer and a lower
            # learning rate warmed up again from zero.
            trainer.save_model(str(out / 'phase1'))
            log.info('Saved phase-1 (best) model to %s', out / 'phase1')
            free_trainer(trainer)
            trainer = None

    if args.sft_epoch:
        nested_run = None
        if mlflow_enabled() and not args.skip_phase1:
            # A nested run, so phase 2's step numbers don't overwrite
            # phase 1's in the parent run.
            import mlflow
            nested_run = mlflow.start_run(run_name=f'{out.name}-sft',
                                          nested=True)
        log.info('Phase 2: sft data only, %g epoch(s) at lr %g, starting '
                 'from %s', args.sft_epochs, args.sft_epoch_learning_rate,
                 args.base_model if args.skip_phase1 else 'phase 1')
        trainer = make_trainer(
            args, model, tokenizer, lang_ids, sft_ds, sft_sampler, val_ds,
            dev_pairs, checkpoints_dir=out / 'checkpoints_sft',
            learning_rate=args.sft_epoch_learning_rate,
            num_train_epochs=args.sft_epochs,
            warmup_steps=args.sft_epoch_warmup_steps,
            eval_steps=args.sft_epoch_eval_steps,
            select_best=False, run_name=f'{out.name}-sft',
            is_final_phase=True)
        trainer.train()
        if torch.cuda.is_available():
            log.info('Phase 2 peak GPU memory: %.1f GB',
                     torch.cuda.max_memory_allocated() / 1e9)
        if nested_run is not None:
            import mlflow
            mlflow.end_run()

    final_dir = out / 'final'
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    with open(final_dir / 'nllb_language_codes.json', 'w') as f:
        codes = dict(LANGS)
        if args.src_lang_dropout:
            codes[UNKNOWN_SOURCE] = constants.NLLB_69_UNKNOWN_SOURCE_CODE
        json.dump(codes, f, indent=2)
    log.info('Saved final model to %s', final_dir)
    if args.hub_model_id:
        trainer.push_to_hub()
    return str(final_dir)


# ---------------------------------------------------------------------------
# Evaluation (metrics replicate sunflower/eval_translation.py)
# ---------------------------------------------------------------------------

_scorers = {}


def compute_sentence_metrics(prediction, reference):
    if not _scorers:
        from sacrebleu.metrics import BLEU, CHRF
        from transformers.models.whisper.english_normalizer import (
            BasicTextNormalizer)
        _scorers.update(normalize=BasicTextNormalizer(),
                        bleu=BLEU(effective_order=True), chrf=CHRF())
    pred = _scorers['normalize'](prediction)
    ref = _scorers['normalize'](reference)
    return {
        'bleu': _scorers['bleu'].sentence_score(pred, [ref]).score,
        'chrf': _scorers['chrf'].sentence_score(pred, [ref]).score / 100.0,
        'len_ratio': len(pred) / max(len(ref), 1),
    }


def compute_direction_metrics(sentence_metrics):
    return {
        'bleu': round(float(np.mean([m['bleu'] for m in sentence_metrics])), 2),
        'chrf': round(float(np.mean([m['chrf'] for m in sentence_metrics])), 4),
        'len_ratio': round(
            float(np.mean([m['len_ratio'] for m in sentence_metrics])), 2),
    }


@torch.no_grad()
def translate(model, tokenizer, lang_ids, texts, src, tgt, batch_size,
              num_beams, max_new_tokens, max_length, length_cap=None):
    '''Translates texts. src is a language code or a list with one code per
    text. tgt likewise; with a list, one batch can mix target languages.
    A source of None (whole argument or per text) means "not given": the
    unknown-source token is used, which needs a model trained with
    --src_lang_dropout.

    One target language: standard NLLB decoding with forced_bos_token_id.
    Several: the target token is given per row as a decoder prefix
    [decoder_start, tgt_lang]. That's identical for greedy decoding, but not
    for beam search, where the forced token counts towards the length
    penalty and the prefix doesn't.

    length_cap: if set, max new tokens per batch is at most
    length_cap * (longest source in batch) + 10, which stops degenerate
    (non-terminating) outputs early.'''
    texts = [normalize_text(t) for t in texts]  # as in training
    n = len(texts)
    src_langs = [src] * n if src is None or isinstance(src, str) else src
    src_langs = [s or UNKNOWN_SOURCE for s in src_langs]
    assert UNKNOWN_SOURCE not in ([tgt] if isinstance(tgt, str) else tgt), (
        'The unknown-source token is not a target language')
    start_id = model.config.decoder_start_token_id
    order = sorted(range(n), key=lambda i: -len(texts[i]))
    outputs = [None] * n
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], return_tensors='pt',
                        padding=True, truncation=True,
                        max_length=max_length).to(model.device)
        enc['input_ids'][:, 0] = torch.tensor(
            [lang_ids[src_langs[i]] for i in idx], device=model.device)
        if isinstance(tgt, str):
            target = {'forced_bos_token_id': lang_ids[tgt]}
        else:
            target = {'decoder_input_ids': torch.tensor(
                [[start_id, lang_ids[tgt[i]]] for i in idx],
                device=model.device)}
        new_tokens = max_new_tokens
        if length_cap:
            new_tokens = min(new_tokens, int(
                length_cap * enc['input_ids'].shape[1]) + 10)
        generated = model.generate(
            **enc, **target, num_beams=num_beams, max_new_tokens=new_tokens,
            max_length=None, use_cache=True)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for i, text in zip(idx, decoded):
            outputs[i] = text.strip()
    return outputs


def evaluate(args, model_path=None):
    model_path = model_path or args.model or str(Path(args.output_dir) / 'final')
    name = args.run_name or Path(model_path.rstrip('/')).name
    if name == 'final':
        name = Path(model_path.rstrip('/')).parent.name
    log.info('Evaluating %s on salt-69', model_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to(
            'cuda' if torch.cuda.is_available() else 'cpu').eval()
    lang_ids = constants.nllb_69_language_token_ids(tokenizer)

    ds = datasets.load_dataset(EVAL_DATA[0], EVAL_DATA[1], split=EVAL_DATA[2])
    languages = args.languages or sorted(
        c[:-5] for c in ds.column_names
        if c.endswith('_text') and c != 'eng_text')

    results = {}
    for mode in args.source_modes:
        results[mode] = evaluate_salt69(
            args, model, tokenizer, lang_ids, ds, languages, model_path,
            name=name if mode == 'given' else f'{name}_nosrc',
            unknown_source=mode == 'unknown')
    if len(results) == 2:
        given = results['given']['per_language']
        unknown = results['unknown']['per_language']
        table = pd.DataFrame([
            {'lang': lang, 'name': given[lang]['name'],
             'eng->xx': given[lang]['eng_to_xx']['chrf'],
             'eng->xx nosrc': unknown[lang]['eng_to_xx']['chrf'],
             'xx->eng': given[lang]['xx_to_eng']['chrf'],
             'xx->eng nosrc': unknown[lang]['xx_to_eng']['chrf'],
             'avg change': round(unknown[lang]['avg_chrf']
                                 - given[lang]['avg_chrf'], 4)}
            for lang in given]).sort_values('avg change')
        log.info('chrF with the source language given vs not given (nosrc):'
                 '\n%s', table.to_string(index=False))
    return results


def evaluate_salt69(args, model, tokenizer, lang_ids, ds, languages,
                    model_path, name, unknown_source):
    '''One salt-69 pass, translating with the source language given, or with
    the unknown-source token (unknown_source). Results are saved under name,
    and MLflow metrics get the prefix nosrc_ when the source isn't given.'''
    log.info('salt-69 with the source language %s',
             'not given' if unknown_source else 'given')
    results = {'per_language': {}}
    predictions, direction_metrics = [], []
    for lang in languages:
        rows = [(e, x) for e, x in zip(ds['eng_text'], ds[f'{lang}_text'])
                if e and x]
        if args.eval_limit:
            rows = rows[:args.eval_limit]
        eng_texts, lang_texts = [r[0] for r in rows], [r[1] for r in rows]
        per_dir = {}
        for direction, src, tgt, sources, refs in (
                ('eng_to_xx', 'eng', lang, eng_texts, lang_texts),
                ('xx_to_eng', lang, 'eng', lang_texts, eng_texts)):
            preds = translate(model, tokenizer, lang_ids, sources,
                              None if unknown_source else src, tgt,
                              batch_size=args.eval_batch_size,
                              num_beams=args.num_beams,
                              max_new_tokens=args.max_new_tokens,
                              max_length=args.max_length)
            sentence = [compute_sentence_metrics(p, r)
                        for p, r in zip(preds, refs)]
            per_dir[direction] = compute_direction_metrics(sentence)
            direction_metrics.append(per_dir[direction])
            for s, p, r, m in zip(sources, preds, refs, sentence):
                predictions.append({
                    'source_lang': src, 'target_lang': tgt, 'source': s,
                    'prediction': p, 'reference': r,
                    'chrf': round(m['chrf'], 4)})
        results['per_language'][lang] = {
            'name': LANGUAGE_NAMES.get(lang, lang),
            'n_examples': len(rows),
            'eng_to_xx': per_dir['eng_to_xx'],
            'xx_to_eng': per_dir['xx_to_eng'],
            'avg_chrf': round((per_dir['eng_to_xx']['chrf']
                               + per_dir['xx_to_eng']['chrf']) / 2, 4),
        }
        log.info('  %s (%s): chrF e2x=%.4f x2e=%.4f', lang,
                 LANGUAGE_NAMES.get(lang, lang), per_dir['eng_to_xx']['chrf'],
                 per_dir['xx_to_eng']['chrf'])

    results['aggregate'] = {
        'avg_bleu': round(float(np.mean([m['bleu'] for m in direction_metrics])), 2),
        'avg_chrf': round(float(np.mean([m['chrf'] for m in direction_metrics])), 4),
        'avg_len_ratio': round(
            float(np.mean([m['len_ratio'] for m in direction_metrics])), 2),
        'n_languages': len(results['per_language']),
    }

    out = Path(args.output_dir) / 'eval'
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / f'{name}_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out / f'{name}_predictions.jsonl', 'w', encoding='utf-8') as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    table = pd.DataFrame([
        {'lang': lang, 'name': r['name'], 'eng->xx': r['eng_to_xx']['chrf'],
         'xx->eng': r['xx_to_eng']['chrf'], 'avg': r['avg_chrf']}
        for lang, r in results['per_language'].items()
    ]).sort_values('avg', ascending=False)
    log.info('salt-69 chrF for %s:\n%s', name, table.to_string(index=False))
    log.info('Aggregate: %s', results['aggregate'])
    log.info('Results saved to %s', results_path)

    if mlflow_enabled():
        # Logs into the training run when called from `all`.
        import mlflow
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=f'eval-{name}')
        mlflow.log_param('eval_model', model_path)
        metrics = {'salt69_avg_chrf': results['aggregate']['avg_chrf'],
                   'salt69_avg_bleu': results['aggregate']['avg_bleu']}
        for lang, r in results['per_language'].items():
            for metric in ('chrf', 'bleu'):
                metrics[f'{metric}_eng_to_{lang}'] = r['eng_to_xx'][metric]
                metrics[f'{metric}_{lang}_to_eng'] = r['xx_to_eng'][metric]
        if unknown_source:
            metrics = {f'nosrc_{k}': v for k, v in metrics.items()}
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(results_path))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('command', choices=['prepare', 'train', 'eval', 'all'])
    p.add_argument('--output_dir', required=True)
    p.add_argument('--base_model', default=BASE_MODEL)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_length', type=int, default=512,
                   help='Max source/target tokens; longer training pairs are '
                        'dropped, eval inputs truncated')

    g = p.add_argument_group('prepare')
    g.add_argument('--max_per_direction', type=int, default=6000,
                   help='Cap on parallel_text pairs per eng->xx / xx->eng')
    g.add_argument('--non_eng_fraction', type=float, default=0.1,
                   help='Non-English pairs, as a fraction of the English-pivot '
                        'parallel_text pairs')
    g.add_argument('--max_per_non_eng_pair', type=int, default=2000)
    g.add_argument('--sft_upsample', type=int, default=2)
    g.add_argument('--limit_rows', type=int,
                   help='Use only the first N parallel_text rows (testing)')

    g = p.add_argument_group('train')
    # Defaults benchmarked on one RTX 5090 (32GB): 3072-token batches without
    # gradient checkpointing ran at ~4,600 source tokens/s with a 27.6GB peak
    # (the worst-case batches run first). 8192 tokens with checkpointing:
    # ~4,150/s, 27.9GB. Fixed batches of 8 pairs: ~1,700/s.
    g.add_argument('--max_tokens_per_batch', type=int, default=3072,
                   help='Token-budget batching: max (pairs x (longest source '
                        '+ longest target)) per micro-batch. 0 = fixed-size '
                        'batches of --per_device_train_batch_size')
    g.add_argument('--max_sequences_per_batch', type=int, default=256)
    g.add_argument('--gradient_checkpointing',
                   action=argparse.BooleanOptionalAction, default=False,
                   help='Needed for token budgets above ~4k on 32GB')
    g.add_argument('--per_device_train_batch_size', type=int, default=8,
                   help='Only used with --max_tokens_per_batch 0')
    g.add_argument('--per_device_eval_batch_size', type=int, default=16)
    g.add_argument('--gradient_accumulation_steps', type=int, default=12,
                   help='Micro-batches per optimizer step. 12 x 3072-token '
                        'batches is ~28k real tokens, like 256 pairs; use '
                        '~32 with fixed batches of 8')
    g.add_argument('--learning_rate', type=float, default=1e-4)
    g.add_argument('--weight_decay', type=float, default=0.0)
    g.add_argument('--warmup_steps', type=int, default=100)
    g.add_argument('--num_train_epochs', type=float, default=2,
                   help='Epochs for phase 1 (all data). Use 1 with '
                        '--sft_epoch')
    g.add_argument('--max_steps', type=int, default=-1)

    g = p.add_argument_group(
        'train: second epoch on sft data only',
        'After phase 1 on all data, optionally train a second phase on just '
        'the sunflower-posttrain sft pairs, which are quality-filtered and '
        'in modern usage, so they rather than the noisier parallel_text '
        'shape the final output. Phase 2 reuses the sft rows of the same '
        'prepared dataset.')
    g.add_argument('--sft_epoch', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='Run the second, sft-only phase')
    g.add_argument('--skip_phase1', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='Skip phase 1 and run only the sft phase, starting '
                        'from --base_model. Use to add an sft phase to a '
                        'model already trained on all data. Implies '
                        '--sft_epoch')
    g.add_argument('--sft_epochs', type=float, default=1.0,
                   help='Epochs for phase 2')
    g.add_argument('--sft_epoch_learning_rate', type=float, default=2e-5,
                   help='Phase-2 learning rate. Lower than phase 1, so the '
                        'short sft-only phase shifts the output register '
                        'without undoing phase 1')
    g.add_argument('--sft_epoch_warmup_steps', type=int, default=25)
    g.add_argument('--sft_epoch_eval_steps', type=int, default=100,
                   help='Phase 2 is only a few hundred steps, so it needs a '
                        'smaller value than --eval_steps')
    g.add_argument('--eval_steps', type=int, default=500,
                   help='Evaluate (dev loss + chrF) and save a checkpoint '
                        'every N optimizer steps')
    g.add_argument('--eval_on_start', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Evaluate the base model at step 0 as a reference')
    g.add_argument('--logging_steps', type=int, default=10)
    g.add_argument('--save_total_limit', type=int, default=3)
    g.add_argument('--early_stopping_patience', type=int, default=6)
    g.add_argument('--dev_chrf_batch_size', type=int, default=64,
                   help='Generation batch size for dev-set chrF at each '
                        'evaluation (64: ~1 min, ~23GB peak during training)')
    g.add_argument('--best_metric', choices=['chrf', 'chrf_nosrc', 'loss'],
                   default='chrf',
                   help='Dev-set metric for best checkpoint and early '
                        'stopping (chrf_nosrc: source language not given; '
                        'needs --src_lang_dropout)')
    g.add_argument('--src_lang_dropout', type=float, default=0.25,
                   help='Fraction of training pairs whose source-language '
                        'token is replaced by the unknown-source token, so '
                        'the source language is optional at inference. 0 = '
                        'always give it (as in earlier runs)')
    g.add_argument('--hub_model_id', help='Push the final model here (private)')

    g = p.add_argument_group('eval')
    g.add_argument('--model', help='Model to evaluate (default: '
                                   'OUTPUT_DIR/final)')
    g.add_argument('--languages', nargs='+',
                   help='Languages to evaluate (default: all in salt-69)')
    g.add_argument('--eval_limit', type=int,
                   help='Max sentences per language (testing)')
    g.add_argument('--eval_batch_size', type=int, default=32)
    g.add_argument('--num_beams', type=int, default=4)
    g.add_argument('--max_new_tokens', type=int, default=256)
    g.add_argument('--run_name', help='Name for result files')
    g.add_argument('--source_modes', nargs='+', choices=['given', 'unknown'],
                   help='Evaluate with the source language given, not given '
                        '(unknown-source token; results saved as '
                        'NAME_nosrc_*), or both. Default: both if '
                        '--src_lang_dropout > 0, else given')
    args = p.parse_args()
    if args.best_metric == 'chrf_nosrc' and not args.src_lang_dropout:
        p.error('--best_metric chrf_nosrc needs --src_lang_dropout > 0')
    if args.skip_phase1:
        # Only phase 2 runs, so there is nothing to select between.
        args.sft_epoch = True
    if args.source_modes is None:
        args.source_modes = (['given', 'unknown'] if args.src_lang_dropout
                             else ['given'])
    # Deduplicate, keeping 'given' first.
    args.source_modes = [m for m in ('given', 'unknown')
                         if m in args.source_modes]
    return args


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    transformers.set_seed(args.seed)
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'train':
        train(args)
    elif args.command == 'eval':
        evaluate(args)
    else:
        if (Path(args.output_dir) / 'pairs').exists():
            log.info('Reusing existing pairs in %s/pairs', args.output_dir)
        else:
            prepare(args)
        final_dir = train(args)
        evaluate(args, model_path=final_dir)
    if mlflow_enabled():
        import mlflow
        mlflow.end_run()


if __name__ == '__main__':
    main()
