"""GRPO fine-tuning of a salt-69 NLLB model on labelled translation pairs.

Takes a model already fine-tuned by
translation_nllb_training_69_languages.py and continues it with
reinforcement learning, to reduce the two failure modes supervised training
leaves behind: repetition loops on low-resource targets, and output drifting
into the wrong language (most often the European language whose NLLB token
was repurposed).

The algorithm is GRPO with no KL term and one optimisation pass per set of
rollouts, which reduces exactly to group-relative REINFORCE: for each source
sentence, sample --num_generations translations, score each one, subtract the
group mean to get an advantage, and take a policy-gradient step. There is no
reference model, no PPO clipping and no importance-sampling ratio, because at
beta=0 with one pass the ratio is identically 1. Nothing here needs TRL,
whose GRPOTrainer is decoder-only (it concatenates prompt and completion into
one causal sequence and slices the completion off the right-hand end, which
has no meaning for an encoder-decoder).

Two kinds of training row, and the reward differs between them.

A *labelled* row carries a reference translation (7,535 of 10,734) and is
scored directly against it. A *round-trip* row does not (3,199, all eng->X):
it carries expected_round_trip_text instead, and is scored by translating the
rollout back to English with a frozen copy of the starting checkpoint and
comparing that with the English reference. The round-trip rows are the only
data for 7 languages -- ber, dag, dga, fra, kab, kln, luy -- and roughly
double another 5 that have 42 labelled rows each, so they are where this run
reaches the weak tail at all.

Reward on a labelled row (summed, weights configurable):

- chrF against the reference translation, in [0, 1]. chrF++ (word_order=2)
  with whitespace counted, matching sunflower/rl_rewards.py's
  chrf_pair_scores rather than the eval metric in the SFT script -- see
  chrf_scores.
- Repetition: 0.0 normally, EXCESSIVE_REPETITION_SCORE for a completion whose
  word trigrams or 20-character n-grams repeat past REPETITION_THRESHOLD.
- Length: 0.0 while the completion is within TRANSLATION_LENGTH_BAND of the
  reference's length, tapering to LENGTH_VIOLATION_SCORE outside it.

A round-trip row keeps those three and adds two more, both of which are
load-bearing rather than refinements:

- Copy: the fraction of the English source's word trigrams reproduced
  verbatim in the rollout, scaled by COPY_PENALTY_SCALE. A rollout that
  simply echoes its English source back-translates perfectly and would score
  ~1.0 chrF for having translated nothing.
- Language: -(1 - p(target language)) from a 62MB classifier, averaged over
  overlapping windows of the rollout. Round-trip chrF is blind to the
  wrong-language failure and on four of these languages actively rewards it:
  salt-69 repurposes NLLB tokens, so dag/dga/kln/luy borrow nld_Latn
  (Dutch), swe_Latn (Swedish), cat_Latn (Catalan) and pol_Latn (Polish). Ask
  for Dagaare, get Swedish, back-translate swe->eng at which NLLB is
  excellent, and the English matches almost exactly for a high reward.

Kalenjin is excluded from the branch by default, being the one language that
is both reward-inverted and outside the classifier's label set; see
--roundtrip_exclude_languages.

All of these are pure penalties, and they are summed and clamped to
MAX_COMBINED_PENALTY before chrF is added, so one rollout can lose at most
one penalty's worth however many detectors it trips, and the total sits in
[-1, 1]. That clamp is what makes four penalties safe alongside two: a
round-trip rollout that copies its source, runs long and drifts out of the
language has tripped three detectors on one underlying fault, and is charged
once. chrF is the only thing that pays: a reward near 0.4 reads as "a decent
translation that broke no rules"; anything negative means a penalty fired. This differs
from sunflower/rl_rewards.py, where the same two rewards pay +1.0 for
acceptable output and floor at -1.0. The floors are identical; the ranges are
not, so each penalty here is half as wide as it is there. Deliberate: paying
every rollout for not repeating itself cancels inside a group anyway (GRPO
subtracts the group mean), so the bonus only widened the gap in the mixed
groups while making a reward of 2.0 mean "nothing went wrong" instead of
"this is a good translation".

A looping completion still loses chrF *and* takes the clamped penalty, which
is the whole mechanism: against a clean rollout that is a swing of about 1.3
(chrF ~0.35 + 1.0), where the chrF spread within a clean group is ~0.05. The
gradient is therefore still dominated by suppressing degeneracy, which is
what this run is for; the clamp halves that dominance rather than removing
it.

The back-translator is a frozen copy of the starting checkpoint, not the
policy. The reward then means the same thing at the last step as at the
first, and -- the reason that matters -- the policy cannot co-adapt with it.
With shared weights, "translate to Dagaare" and "translate back to English"
are the same parameters, and a private encoding that round-trips perfectly
while being no language at all would score 1.0 on every round-trip row. That
is the classic degenerate solution to a round-trip objective; freezing the
back-translator removes it rather than penalising it. Under --lora_r the
frozen model is the policy with the adapter switched off, so nothing is
loaded twice.

Reference normalisation matters and is easy to get wrong: the model is
trained on normalize_text output and emits ASCII punctuation only, while the
dataset's references carry typographic punctuation. Scoring chrF against raw
references would penalise correct translations for every curly apostrophe.
filter_pairs (reused from the SFT script) applies normalize_text to both
sides, so sources and references here go through exactly what training did.

Evaluation during training is greedy chrF via the SFT script's dev_chrf,
on one of two dev sets (--dev_source). Neither is free:

  sft (default): the sft_translations dev split, 1,080 pairs, the same set the
    supervised script reports chrF on -- so the two runs' dev curves are
    comparable and salt-69 stays untouched. It covers 65 of the 74 non-English
    salt-69 languages, with 3-8 pairs for most of them, and is missing Berber,
    Dagbani, Dagaare, French, Kabyle, Kalenjin, Ikposo and Luhya.

  rl: the rl_translations dev split, 1,380 pairs, a flat 10 per direction,
    covering 69 of the 74. It is drawn from salt-69 -- all 1,380 rows match
    salt69_texts -- because for several of these languages salt-69 is the only
    reliable reference data in existence. Selecting checkpoints on it means
    the final salt-69 score is selected-on rather than held out. The script
    says so loudly at startup and skips the leakage filter for it.

Which to pick depends on the stage. Every language the sft split lacks is
also a language with no labelled training data (see below), so for this
reference-only phase its coverage gap costs nothing in detecting improvement
-- only in detecting regression, which the final salt-69 eval catches. Once
the round-trip rows are in play, 7 of those 9 become trainable and the gap
starts to matter.

The rl_translations *train* split is clean either way: 1 row of 10,734
touches salt-69, and filter_pairs drops it.

What this phase can and cannot reach: of the 74 non-English salt-69
languages, 13 have no labelled train row at all -- Berber, Dagbani, Dagaare,
French, Kabyle, Kalenjin, Luhya (87-88 round-trip rows each, so trainable
only at the next stage) and Ikposo, Thur, Lulubo, Kebu, Sebei, So (no
rl_translations data of either kind). A further 29 have fewer than 50,
including the repetition-prone ones the run is aimed at: Lendu, Aringa,
Kupsabiny, Kwamba and Ik have 42 labelled prompts each, while Dagaare and
Kalenjin have none. Expect this phase to move the mid-tail, and expect the
weakest languages to need the round-trip stage.

The best checkpoint by dev chrF is kept and early stopping is on by
default: published GRPO runs on NLLB report late collapse as the reward
spread within groups shrinks towards zero, so the last checkpoint is not
safe to assume. Watch rollouts/zero_variance_groups -- when it approaches
1.0, every group's rollouts agree, the advantages are zero and the run has
stopped learning. Raise --temperature or stop.

The final evaluation on Sunbird/salt-69 is the SFT script's, unchanged, so
scores compare directly with the supervised runs and with
/sunflower/scripts/eval/translation.py.

Requires torch, transformers, datasets, sacrebleu, sacremoses, torchao, and
peft only for --lora_r > 0.

Usage (from the repo root, in /venv/main):
  # continue a supervised checkpoint with RL, then evaluate on salt-69
  python notebooks/training/translation_nllb_grpo_69_languages.py all \
      --output_dir /workspace/nllb-1.3b-grpo \
      --base_model /workspace/nllb-1.3b-sft/final

  # smoke test: 5 steps, small dev slice, no MLflow
  python notebooks/training/translation_nllb_grpo_69_languages.py train \
      --output_dir /tmp/grpo-smoke \
      --base_model /workspace/nllb-1.3b-sft/final \
      --max_steps 5 --dev_limit 64 --eval_steps 5 --no-mlflow

  # evaluate an already-trained GRPO model on salt-69
  python notebooks/training/translation_nllb_grpo_69_languages.py eval \
      --output_dir /workspace/nllb-1.3b-grpo

MLflow logging is required by default, as in the SFT script: train/eval/all
stop before doing any work unless https://mlflow.sunbird.ai accepts the
credentials in MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD. Runs
land in the "translation-grpo" experiment, separate from the SFT script's
"translation" -- these runs report the same eval metric names but mean
something different by a step, so mixing the two in one experiment makes the
metric charts unreadable. The server and the experiment are both set in the
scripts and ignore MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME. Pass
--no-mlflow to run without logging.
"""

import argparse
import collections
import contextlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import constants  # noqa: E402
# The supervised script is imported, not copied: normalize_text, the salt-69
# leakage filter, pair filtering, tokenisation/length filtering, dev chrF, the
# salt-69 evaluation and the MLflow preflight are all shared. It guards main()
# and does nothing at import beyond setting PYTORCH_CUDA_ALLOC_CONF, so
# importing it is safe.
import translation_nllb_training_69_languages as sft  # noqa: E402

log = logging.getLogger('nllb69-grpo')

MLFLOW_EXPERIMENT = 'translation-grpo'

RL_DATA = ('Sunbird/sunflower-posttrain-data', 'rl_translations')
ORIGIN = 'rl'
ORIGIN_ROUNDTRIP = 'rl-roundtrip'  # filter_pairs only dedups/length-filters 'parallel*' origins


# ---------------------------------------------------------------------------
# Rewards
#
# Deliberately self-contained rather than imported from
# /sunflower/sunflower/rl_rewards.py. The repo dependency already runs
# sunflower -> salt (sunflower/data/pretraining.py lazily imports salt), so
# importing it back here would invert that, and these three rewards are ~110
# lines of strings-and-sets code with no shared state. The constants below are
# spelled as they are there, so checking for drift is a diff rather than a
# reading exercise. The rewards that do justify the import -- language
# identification and the round-trip branch -- arrive with the next stage.
# ---------------------------------------------------------------------------

# Both floors sit in the same range as the chrF they are summed with: under
# any advantage scaling every group is divided by the same number, so a reward
# that can reach -5 simply outweighs everything else in the batch. Same values
# as sunflower/rl_rewards.py, but reached from a 0.0 plateau rather than a
# +1.0 one -- see the module docstring.
LENGTH_VIOLATION_SCORE = -0.5
EXCESSIVE_REPETITION_SCORE = -1.0
# Floor of the language penalty, reached when the classifier gives the target
# language probability 0 across every window of the rollout. Same magnitude as
# the other two floors so no single detector can outvote the rest.
LANGUAGE_DRIFT_SCORE = -1.0  # flat charge for drift; see language_scores
LANGID_TOP_K = 5

# The most the two penalties can take off together. They detect overlapping
# failures -- a non-terminating rollout trips both, and so does an empty one
# -- so summing them unclamped charges one fault twice and lets a single
# rollout reach -2.0 when the entire chrF range is 1.0. Clamped here to one
# penalty's worth, so the pair at full strength can at most cancel a perfect
# translation. Set --max_penalty 2.0 to recover the uncapped sum.
#
# The cap is applied first and --floor_reward then clips the total at 0. An
# earlier version of this comment argued against the floor on the grounds
# that it would flatten most groups; measuring it on 13,800 real outputs
# showed otherwise, and the argument was wrong on both counts. It zeroes
# 20.7% of rollouts, not the ~52% that trip some penalty, because a small
# length penalty does not cancel a decent chrF. And it empties a whole group
# only 4.6% of the time at G=8, 3.8% at G=12, concentrated in Lendu, Ikposo
# and Berber where ~98% of rollouts are useless anyway.
#
# The ordering lost among floored rollouts -- "this repetition loop is closer
# to the reference than that one" -- is noise, not signal: a loop and a
# wrong-language reply are equally worthless, and propagating a gradient
# between them only adds variance. rollouts_zero_variance_groups is the
# metric to watch; under the floor it counts groups with no usable rollout
# rather than groups whose rollouts happen to tie.
MAX_COMBINED_PENALTY = 1.0

# Length ratio (completion / reference characters) that costs nothing, and the
# ratio distance beyond it over which the penalty reaches its floor. The band's
# width is set by how much a *correct* translation's length legitimately
# varies: measured against a single expansion factor that residual has sd
# 0.16, so a taper of 0.4 means a rollout has to be ~2.5 sd outside normal
# variation before it takes the full penalty.
TRANSLATION_LENGTH_BAND = (0.9, 1.1)
TRANSLATION_LENGTH_TAPER = 0.4

REPETITION_NGRAM_SIZE = 3
REPETITION_CHAR_NGRAM_SIZE = 20
REPETITION_THRESHOLD = 0.2

_chrf = None


def chrf_scores(hypotheses, references):
    '''Pairwise sentence chrF in [0, 1].

    word_order=2 (chrF++, so word n-grams count too) and whitespace=True,
    matching sunflower/rl_rewards.py's chrf_pair_scores. Both differ from the
    SFT script's compute_sentence_metrics, which is plain chrF over
    BasicTextNormalizer output -- that one has to stay as it is to keep the
    salt-69 numbers comparable with earlier runs, whereas the reward wants
    credit for word order and no normalisation standing between the policy and
    what it is being scored on.
    '''
    global _chrf
    if _chrf is None:
        from sacrebleu.metrics import CHRF
        # Built without references=, so sentence_score never mutates it.
        _chrf = CHRF(word_order=2, whitespace=True)
    return [_chrf.sentence_score(h, [r]).score / 100.0
            for h, r in zip(hypotheses, references)]


def _ngram_repetition_ratio(units, size):
    '''Fraction of n-grams over `units` that are repeats, or None if there are
    fewer than `size` units to form one.'''
    if len(units) < size:
        return None
    ngrams = set()
    total = 0
    for ngram in zip(*[units[i:] for i in range(size)]):
        ngrams.add(ngram)
        total += 1
    return 1 - len(ngrams) / total


def repetition_scores(texts):
    '''EXCESSIVE_REPETITION_SCORE for texts that repeat themselves, else 0.0.

    Two detectors, because they catch different failures. Word trigrams find
    a repeated word or phrase. Character 20-grams find repetition with no word
    boundaries at all -- a run of punctuation, or one character repeated, reads
    as a single enormous "word" under a whitespace split, so the word-level
    detector never sees it.
    '''
    scores = []
    for text in texts:
        body = text.lower()
        if not body:
            # Empty is punished by the length reward, which is the failure it
            # actually is; charging it here as well would double-count.
            scores.append(0.0)
            continue
        ratios = [_ngram_repetition_ratio(body.split(), REPETITION_NGRAM_SIZE),
                  _ngram_repetition_ratio(body, REPETITION_CHAR_NGRAM_SIZE)]
        excessive = any(r is not None and r > REPETITION_THRESHOLD
                        for r in ratios)
        scores.append(EXCESSIVE_REPETITION_SCORE if excessive else 0.0)
    return scores


def _length_band_score(ratio):
    '''0.0 inside TRANSLATION_LENGTH_BAND, tapering linearly to
    LENGTH_VIOLATION_SCORE one taper outside it.'''
    low, high = TRANSLATION_LENGTH_BAND
    if low <= ratio <= high:
        return 0.0
    distance = (low - ratio) if ratio < low else (ratio - high)
    return max(LENGTH_VIOLATION_SCORE,
               -distance / TRANSLATION_LENGTH_TAPER)


def length_scores(texts, expected_lengths):
    '''Length penalty from the ratio of each text to the length expected of
    a correct translation.

    On a labelled row that expectation is the reference translation's own
    length, so the ratio is script-neutral and the band means the same thing
    in every language -- which is what makes one band usable across 69 target
    languages in four scripts. A round-trip row has no target-language
    reference, so score_rollouts passes the English source length scaled by
    translation_length_expansion instead. A taper rather than a cliff: a step
    function gives no gradient to a rollout already outside the band.

    This is where a non-terminating rollout is punished hardest. It runs to
    the generation cap, so its ratio is several times 1.0, which puts it on
    the floor whatever its chrF does.
    '''
    scores = []
    for text, expected in zip(texts, expected_lengths):
        body = text.strip()
        if not body or expected <= 0:
            scores.append(LENGTH_VIOLATION_SCORE)
            continue
        scores.append(_length_band_score(len(body) / expected))
    return scores


# ---------------------------------------------------------------------------
# Round-trip rewards: the copy penalty, the length expectation against
# English, and language identification. Ported from sunflower/rl_rewards.py
# for the reason above -- and because the dependency would have to run
# backwards, since sunflower/data/pretraining.py already imports salt. What is
# ported is the calibration, not the code: every number here was measured on
# this corpus or replayed over a real run, and the constants are spelled as
# they are there so checking for drift is a diff rather than a reading
# exercise.
# ---------------------------------------------------------------------------

# --- Copy penalty ---
#
# The exploit this exists to close. A round-trip row is scored by translating
# the rollout back to English, so a rollout that simply echoes its English
# source back-translates perfectly and scores ~1.0 chrF for having translated
# nothing. NLLB will do this readily on a language it is unsure of.

# Nigerian Pidgin is exempt: it is English-lexified, so reproducing English
# wholesale is correct translation there rather than a failure to translate.
# Measured over this corpus's human references, 9 of the 10 highest-copying
# rows are Pidgin, whose content-word retention averages 0.47 against a 0.03
# corpus-wide mean. Exempting it removes the dominant false-positive
# population without needing a per-language calibration table.
COPY_EXEMPT_LANGUAGES = frozenset({'pcm'})

# Trigrams, not single words. Borrowing a term with no local equivalent --
# "photosynthesis", "wifi" -- is what a translator legitimately does, and it
# is one token long; three consecutive source words surviving verbatim is not
# borrowing but an untranslated span. Over the human reference translations
# the trigram rate is 0.0000 (p99 0.000, 0.2% of rows above zero), which is
# why the penalty needs no tolerance threshold. At bigrams 3.5% of human rows
# would be penalised; at 4-grams the human rate is also zero but a third of
# the signal is lost.
COPY_NGRAM_SIZE = 3

# Below this many source trigrams the rate is too coarse to act on -- a single
# match in a three-trigram source would read as 0.33 -- so short and
# name-dense sources are left alone.
COPY_MIN_SOURCE_NGRAMS = 10

# The raw rate is too weak to correct the behaviour it is aimed at. Replayed
# over one run's round-trip rollouts, the branch mean still climbed +0.195
# with the rate subtracted as-is, so copying kept paying; 3x is roughly
# break-even and 5x turns the drift to -0.118, i.e. the policy is pushed back
# rather than merely slowed. Human reference translations are unaffected at
# any of these scales -- the worst of 529 of them takes 0.046 at 5x -- because
# the human trigram-copy floor is genuinely zero rather than merely small.
COPY_PENALTY_SCALE = 5.0

# sunflower also caps this penalty at 1.0 individually (COPY_PENALTY_MAX).
# Not ported: MAX_COMBINED_PENALTY already clamps the sum of the penalties at
# the same value, and since every penalty here is <= 0, pre-capping one of
# them cannot change max(-1, sum). One clamp, in one place.


def copy_ngrams(text, size=COPY_NGRAM_SIZE):
    '''`size`-grams over the words of `text` made entirely of lowercase
    ASCII a-z.

    Deliberately the crudest filter that works: a word carrying a capital, an
    accent, a digit or attached punctuation is dropped, which removes proper
    nouns, dates, figures and acronyms -- the things a translation is supposed
    to reproduce verbatim -- with no sentence-initial tracking and no
    per-language capitalisation rules, both fragile across this language set.
    It under-detects rather than over-detects, the right direction for a
    penalty.

    Dropped words are bridged rather than breaking the sequence, so an n-gram
    can span a name. That is the simple choice rather than the faithful one:
    it can form n-grams that were never contiguous, which reads slightly high
    on name-dense text.
    '''
    words = [w for w in text.split()
             if w.isascii() and w.isalpha() and w.islower()]
    return [tuple(words[i:i + size]) for i in range(len(words) - size + 1)]


def copied_ngram_rate(source, hypothesis):
    '''Fraction of the source's trigrams reproduced verbatim in
    `hypothesis`, or None when the source yields fewer than
    COPY_MIN_SOURCE_NGRAMS of them. None means "not assessable", which
    callers treat as no penalty rather than as a rate of 0.0.'''
    source_ngrams = copy_ngrams(source)
    if len(source_ngrams) < COPY_MIN_SOURCE_NGRAMS:
        return None
    hypothesis_ngrams = set(copy_ngrams(hypothesis))
    return sum(1 for n in source_ngrams if n in hypothesis_ngrams) / len(
        source_ngrams)


def copy_scores(sources, hypotheses, target_languages, apply_to):
    '''Penalty for leaving the English source untranslated, 0.0 to -inf
    before the combined clamp. Only the rows flagged in `apply_to` are
    charged; a labelled row is already scored against a target-language
    reference, which punishes copying directly.'''
    scores = []
    for source, hypothesis, language, applies in zip(
            sources, hypotheses, target_languages, apply_to):
        if not applies or language in COPY_EXEMPT_LANGUAGES:
            scores.append(0.0)
            continue
        rate = copied_ngram_rate(source, hypothesis)
        scores.append(0.0 if rate is None else -COPY_PENALTY_SCALE * rate)
    return scores


# --- Length expectation for a row with no target-language reference ---
#
# A round-trip row can only be measured against English
# (expected_round_trip_text, else the source), scaled by how far a translation
# into that language expands its source. Measured over the 4,473 usable
# English->X reference rows of the training split that expansion is 1.08x --
# except in the Ge'ez syllabary, which carries a whole syllable per glyph and
# runs 0.68x. Ge'ez is the only split worth making: decomposing the spread
# across 57 non-Ge'ez languages gives a between-language sd of 0.110 against a
# within-language, sentence-to-sentence sd of 0.148, so even a perfect
# per-language table would shrink the total spread only from 0.184 to 0.148 --
# a fifth of a standard deviation, for fifty-seven entries whose individual
# medians (30-200 rows each) are themselves noisy.
TRANSLATION_EXPANSION_GEEZ = 0.68
TRANSLATION_EXPANSION_DEFAULT = 1.08

GEEZ_SCRIPT_LANGUAGES = frozenset({'amh', 'tir'})


def translation_length_expansion(target_language):
    '''Expected length of a translation into `target_language` relative to
    its English source.'''
    return (TRANSLATION_EXPANSION_GEEZ
            if target_language in GEEZ_SCRIPT_LANGUAGES
            else TRANSLATION_EXPANSION_DEFAULT)


# --- Language identification ---
#
# Not optional on this branch, and this is the subtle part. Round-trip chrF is
# blind to the wrong-language failure, and on four of these languages it
# actively rewards it: salt-69 repurposes NLLB tokens, and dag/dga/kln/luy
# borrow nld_Latn (Dutch), swe_Latn (Swedish), cat_Latn (Catalan) and
# pol_Latn (Polish). Ask for Dagaare, get Swedish; back-translate swe->eng,
# at which NLLB is excellent, and the result matches the English reference
# almost exactly for a high reward. Emit real Dagaare and NLLB reads it as
# broken Swedish, back-translates badly, and the reward falls. Without this
# penalty the branch pays the policy to abandon the language it is learning.
LANGID_MODEL = 'Sunbird/sunflower_language_classification'

# Sliding windows rather than whole-text scoring. Whole-text is dominated by
# whichever language holds most of the mass, so a rollout that keeps the
# target's morphology and function words while swapping in English content
# words still classifies as the target with high confidence. Scoring
# overlapping windows and averaging lets the substituted spans register:
# measured within-prompt discrimination between a copying rollout and a clean
# one rose from 20% to 49% of pairs (ties 68% -> 29%) moving from whole text
# to 10-word windows.
LANGID_CHUNK_WORDS = 10
LANGID_CHUNK_WORD_STRIDE = 5
# Fallback for text that is not whitespace-delimited: not every language
# separates words with spaces, and a degenerate rollout can arrive as one
# unbroken run. ~5 characters a word, a reasonable analogue of the word
# windows rather than a separately tuned size.
LANGID_CHUNK_CHARS = 50
LANGID_CHUNK_CHAR_STRIDE = 25
LANGID_CHUNK_MAX_WORD_CHARS = 20

LANGID_BATCH_SIZE = 256

_langid = {}


def langid_supported():
    '''Language codes the classifier can detect, read from its config so it
    cannot drift from the checkpoint. Only the config is fetched, not the
    weights, so this is cheap enough to call while building the data.'''
    if 'supported' not in _langid:
        config = transformers.AutoConfig.from_pretrained(LANGID_MODEL)
        _langid['supported'] = frozenset(config.id2label.values())
    return _langid['supported']


def load_language_classifier(device):
    '''Load the classifier (62MB, a T5 sequence classifier) onto `device`.'''
    if 'model' not in _langid:
        log.info('Loading the language classifier %s', LANGID_MODEL)
        _langid['tokenizer'] = transformers.AutoTokenizer.from_pretrained(
            LANGID_MODEL)
        model = transformers.AutoModelForSequenceClassification.\
            from_pretrained(LANGID_MODEL).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _langid['model'] = model
        # Int-keyed: a config loaded from JSON can carry string keys.
        _langid['id2label'] = {int(i): label for i, label
                               in model.config.id2label.items()}
    return _langid


def _sliding_windows(length, size, stride):
    '''Window bounds covering `length` items, with the final window anchored
    at the end. The anchoring matters: a plain range leaves a tail of up to
    size - 1 items in no window at all, which would leave the end of a long
    rollout unscored -- exactly where output that drifts out of the target
    language tends to drift.'''
    if length <= size:
        return [(0, length)]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] + size < length:
        starts.append(length - size)
    return [(start, start + size) for start in starts]


def language_chunks(text):
    '''`text` split into overlapping windows for language scoring.'''
    text = text.strip()
    if not text:
        return []
    words = text.split()
    spaced = (len(words) > 1
              and len(text) / len(words) <= LANGID_CHUNK_MAX_WORD_CHARS)
    if spaced:
        return [' '.join(words[a:b]) for a, b in _sliding_windows(
            len(words), LANGID_CHUNK_WORDS, LANGID_CHUNK_WORD_STRIDE)]
    return [text[a:b] for a, b in _sliding_windows(
        len(text), LANGID_CHUNK_CHARS, LANGID_CHUNK_CHAR_STRIDE)]


def language_probabilities(texts, device):
    '''Classifier probability distributions for `texts`, one dict each.'''
    state = load_language_classifier(device)
    tokenizer, model = state['tokenizer'], state['model']
    id2label = state['id2label']
    out = []
    for start in range(0, len(texts), LANGID_BATCH_SIZE):
        batch = texts[start:start + LANGID_BATCH_SIZE]
        enc = tokenizer(batch, return_tensors='pt', padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            probabilities = model(**enc).logits.float().softmax(-1).cpu()
        out.extend({id2label[i]: float(p) for i, p in enumerate(row)}
                   for row in probabilities)
    return out


def language_scores(texts, target_languages, apply_to, device,
                    top_k=LANGID_TOP_K):
    '''Penalty for drifting out of the target language, gated on a top-k test.

    The classifier's distribution is averaged over the text's chunks and the
    target language looked up in that mean distribution. A rollout whose
    target language is among the top `top_k` labels is charged nothing. The
    penalty is here to catch output that is some other language entirely, not
    to arbitrate between neighbours no classifier can separate: the Ugandan
    Bantu cluster (Runyoro, Rutooro, Rukiga, Runyankole, Ruruuli, Lusoga) is
    routinely confused, and under the ungated score a *correct* Runyoro
    rollout with chrF 0.64 was charged -0.67 because Runyankole ranked above
    it -- spending most of the reward range on a distinction the detector
    cannot make.

    Flat LANGUAGE_DRIFT_SCORE outside the top k, 0.0 inside it -- no grading
    by p(target). The classifier is neither reliable nor calibrated, so the
    size of its probability mass carries no information worth spending reward
    range on; only the rank is worth reading, and only coarsely. Grading by
    it also made the penalty a per-language offset of whatever that language
    happened to score, which is noise dressed as a measurement.

    Only the rows flagged in `apply_to` are charged. A row whose target
    language the classifier has no label for scores 0.0. That should not
    happen: load_rl_prompts drops those rows from the branch rather than
    training on them unguarded.
    '''
    chunks, owners = [], []
    for i, (text, language, applies) in enumerate(
            zip(texts, target_languages, apply_to)):
        if not applies or language not in langid_supported():
            continue
        pieces = language_chunks(text)
        chunks.extend(pieces)
        owners.extend([i] * len(pieces))
    scores = [0.0] * len(texts)
    if not chunks:
        return scores
    summed = collections.defaultdict(collections.Counter)
    for owner, probabilities in zip(owners, language_probabilities(
            chunks, device)):
        summed[owner].update(probabilities)
    for owner, totals in summed.items():
        if not any(language == target_languages[owner]
                   for language, _ in totals.most_common(top_k)):
            scores[owner] = LANGUAGE_DRIFT_SCORE
    return scores


def score_rollouts(texts, references, args, sources=None, target_languages=None,
                   roundtrip=None, back_translations=None, device='cpu'):
    '''Per-rollout reward components and their weighted total.

    Two branches, chosen per rollout by the boolean `roundtrip`. Passing no
    round-trip arguments scores every rollout on the labelled branch, which is
    what the labelled-only phase did.

    labelled
        chrF(rollout, reference) against the target-language reference, with
        the repetition and length penalties. The length band is measured
        against the reference's own length.

    round-trip
        chrF(back-translation, reference), where `references` holds
        expected_round_trip_text (English) and `back_translations` holds the
        rollout translated back to English by the frozen back-translator. Adds
        the copy and language penalties -- see copy_scores and language_scores
        for why neither is optional -- and measures length against the English
        reference scaled by translation_length_expansion, there being no
        target-language reference to measure against.

    The penalties are weighted, summed and clamped to --max_penalty before
    chrF is added, so correlated failures are charged once; see
    MAX_COMBINED_PENALTY. That clamp is what makes the round-trip branch's
    four penalties safe to add to the labelled branch's two: a round-trip
    rollout that copies its source, runs long and drifts out of the language
    trips three detectors on one underlying fault and is still charged once.

    The returned 'penalty' component is that clamped weighted sum; the others
    are the raw per-detector scores behind it, so a sample log reading
    `copy=-1.700 len=-1.000 penalty=-1.000` is the clamp doing its job and
    `reward_penalty` tracks what the policy actually paid.

    Under --floor_reward (the default) the total is then clipped at 0, so
    `chrf=+0.180 penalty=-1.000` logs as a reward of 0.000 rather than
    -0.820. The components still show what happened; only the total is
    clipped. A group where every rollout floors has no gradient, which is the
    intent -- see MAX_COMBINED_PENALTY.
    '''
    n = len(texts)
    roundtrip = (np.zeros(n, dtype=bool) if roundtrip is None
                 else np.asarray(roundtrip, dtype=bool))
    any_roundtrip = bool(roundtrip.any())
    if any_roundtrip and (sources is None or target_languages is None
                          or back_translations is None):
        raise ValueError('round-trip rollouts need sources, target_languages '
                         'and back_translations')

    # What chrF sees: the rollout itself on a labelled row, its
    # back-translation on a round-trip row. Both are compared with
    # `references`, which holds whichever reference the row carries.
    scored = ([back_translations[i] if roundtrip[i] else texts[i]
               for i in range(n)] if any_roundtrip else list(texts))
    chrf = np.array(chrf_scores(scored, references))

    # Every other detector reads the rollout, not the back-translation: they
    # are judging what the policy produced.
    repetition = np.array(repetition_scores(texts))
    expected_lengths = [
        len(reference) * (translation_length_expansion(target_languages[i])
                          if roundtrip[i] else 1.0)
        for i, reference in enumerate(references)]
    length = np.array(length_scores(texts, expected_lengths))

    if any_roundtrip:
        copy = np.array(copy_scores(sources, texts, target_languages,
                                    roundtrip))
        langid = np.array(language_scores(texts, target_languages, roundtrip,
                                          device, args.langid_top_k))
    else:
        copy = np.zeros(n)
        langid = np.zeros(n)

    penalty = np.maximum(
        args.repetition_weight * repetition + args.length_weight * length
        + args.copy_weight * copy + args.langid_weight * langid,
        -abs(args.max_penalty))
    total = args.chrf_weight * chrf + penalty
    if args.floor_reward:
        total = np.maximum(total, 0.0)
    return total, {'chrf': chrf, 'repetition': repetition, 'length': length,
                   'copy': copy, 'langid': langid, 'penalty': penalty}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def finalise_pairs(pairs, tokenizer, lang_ids, eval_texts, args, name,
                   limit=None, seed=42, max_source_tokens=0,
                   filter_fn=None):
    '''Normalise, filter, length-filter and summarise a set of pairs.

    filter_pairs applies normalize_text to both sides and drops empty,
    identical and salt-69-leaked rows; tokenize_and_length_filter drops rows
    over --max_length tokens. Both are the SFT script's, so what reaches
    training here has been through exactly what its training data was.

    Pass eval_texts={} to keep salt-69 rows -- only --dev_source rl does, and
    it warns first.

    `limit` samples rather than truncates: these splits are ordered by
    language, so a head slice covers a handful of languages rather than the
    range being measured.

    filter_fn replaces sft.filter_pairs, for the round-trip rows that its
    identical-pair check would throw away; see filter_roundtrip_pairs.
    '''
    pairs = (filter_fn or sft.filter_pairs)(pairs, eval_texts)
    if not len(pairs):
        raise SystemExit(
            f'{name}: no pairs survived filtering. If they were all dropped '
            'as "in salt-69", this split overlaps the benchmark and cannot '
            'be used.')
    ds = sft.tokenize_and_length_filter(pairs, tokenizer, lang_ids, args, name)
    if max_source_tokens:
        # Rollout cost is dominated by the longest source in the batch, and
        # this corpus mixes single sentences with multi-sentence paragraphs.
        # Training prompts only: capping the dev set would narrow what is
        # being measured rather than make anything cheaper.
        before = len(ds)
        ds = ds.filter(lambda n: n <= max_source_tokens,
                       input_columns='n_source_tokens')
        log.info('%s: dropped %d of %d pairs over %d source tokens',
                 name, before - len(ds), before, max_source_tokens)
    if limit and limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))
        log.info('%s: sampled %d pairs', name, limit)
    rows = ds.select_columns(['source', 'target', 'source.language',
                              'target.language', 'origin']).to_dict()
    directions = list(zip(rows['source.language'], rows['target.language']))
    counts = collections.Counter(t for _, t in directions)
    log.info('%s: %d pairs, %d directions, %d target languages '
             '(median %d rows per target language)', name,
             len(rows['source']), len(set(directions)), len(counts),
             int(np.median(list(counts.values()))))
    return rows


def filter_roundtrip_pairs(pairs, eval_texts):
    '''sft.filter_pairs for round-trip rows: the same normalisation and the
    same salt-69 leakage filter, without the src == tgt drop.

    The "target" of a round-trip row is expected_round_trip_text, the English
    back-translation reference, and it equals the English source in 3,047 of
    the 3,199 rows. That is the normal case rather than a corrupt row, so
    filter_pairs would discard 95% of the branch. Everything else it does
    still applies, which is why this is a near-copy and not a bypass: the
    normalisation has to match (the model emits ASCII punctuation while the
    references carry typographic punctuation, so chrF against an
    unnormalised reference would penalise correct output for every curly
    apostrophe), and the leakage filter has to run (measured: 0 of 3,199
    round-trip sources and 0 of their references are in salt-69 today, but
    nothing stops a future revision of the dataset adding some).
    '''
    keep = sft.Pairs()
    dropped = collections.Counter()
    data = pairs.data
    for src, tgt, sl, tl, origin in zip(
            *(data[c] for c in sft.Pairs.COLUMNS)):
        src, tgt = sft.normalize_text(src), sft.normalize_text(tgt)
        if not src or not tgt:
            dropped['empty'] += 1
            continue
        if src in eval_texts or tgt in eval_texts:
            dropped['in salt-69'] += 1
            continue
        keep.add(sl, tl, src, tgt, origin)
    log.info('Filtered round-trip: kept %d, dropped %s', len(keep),
             dict(dropped))
    return keep


def load_rl_prompts(args, tokenizer, lang_ids, eval_texts):
    '''The prompts GRPO samples from, both kinds, in one row set.

    A row with a target_text is a labelled prompt, scored against that
    reference. A row without one is a round-trip prompt: it carries
    expected_round_trip_text instead, is always eng->X, and is scored by
    translating the rollout back to English. `origin` tells the two apart
    downstream, and a row's branch is fixed here -- nothing later re-derives
    it from whether a field is empty.

    Round-trip rows are kept at their natural rate, ~30% of the corpus. They
    are the only data for 7 languages (ber, dag, dga, fra, kab, kln, luy) and
    roughly double another 5 (gwr, ikx, kdj, led, lsm, at 42 labelled rows
    each), so the branch is where this run can reach the weak tail at all.
    '''
    ds = datasets.load_dataset(*RL_DATA, split='train')
    if args.limit_rows:
        ds = ds.select(range(min(args.limit_rows, len(ds))))
    pairs = sft.Pairs()
    roundtrip = sft.Pairs()
    excluded = collections.Counter()
    unsupported = set()
    for source, target, round_trip, src_lang, tgt_lang in zip(
            ds['source_text'], ds['target_text'],
            ds['expected_round_trip_text'], ds['source_language'],
            ds['target_language']):
        if target:
            pairs.add(src_lang, tgt_lang, source, target, ORIGIN)
            continue
        if not args.roundtrip:
            excluded['--no-roundtrip'] += 1
        elif tgt_lang in args.roundtrip_exclude_languages:
            excluded[f'--roundtrip_exclude_languages ({tgt_lang})'] += 1
        elif args.langid_weight and tgt_lang not in langid_supported():
            # Without a classifier label there is no wrong-language guard,
            # and on this branch that is not a missing nicety: the reward
            # would pay the policy for emitting the European language whose
            # NLLB token this language borrows. Kalenjin (cat_Latn) is the
            # case that prompted the check; anything else that appears here
            # is the same problem and gets the same answer.
            unsupported.add(tgt_lang)
            excluded[f'no classifier label ({tgt_lang})'] += 1
        elif not round_trip and not source:
            excluded['no round-trip reference'] += 1
        else:
            # expected_round_trip_text differs from source_text in 162 of
            # 3,199 rows, so it is the reference even though it usually
            # matches; falling back to the source covers a blank one.
            roundtrip.add(src_lang, tgt_lang, source, round_trip or source,
                          ORIGIN_ROUNDTRIP)
    log.info('rl_translations/train: %d rows, %d labelled, %d round-trip, '
             '%d excluded %s', len(ds), len(pairs), len(roundtrip),
             sum(excluded.values()), dict(excluded))
    if unsupported:
        log.warning('Round-trip rows dropped for %s: the language '
                    'classifier has no label for them, so the '
                    'wrong-language penalty cannot be applied and the '
                    'round-trip reward alone would favour the European '
                    'language whose NLLB token they borrow. Pass '
                    '--langid_weight 0 to train on them unguarded.',
                    ', '.join(sorted(unsupported)))
    rows = finalise_pairs(pairs, tokenizer, lang_ids, eval_texts, args,
                          'train',
                          max_source_tokens=args.max_source_tokens)
    if len(roundtrip):
        extra = finalise_pairs(
            roundtrip, tokenizer, lang_ids, eval_texts, args,
            'train/round-trip', max_source_tokens=args.max_source_tokens,
            filter_fn=filter_roundtrip_pairs)
        rows = {column: rows[column] + extra[column] for column in rows}
        log.info('train: %d prompts in total', len(rows['source']))
    return rows


def load_dev_pairs(args, tokenizer, lang_ids, eval_texts):
    '''The dev set named by --dev_source. See the module docstring for what
    each one covers and what it costs.'''
    pairs = sft.Pairs()
    if args.dev_source == 'sft':
        # Built exactly as the SFT script builds the set it reports chrF on,
        # so the two runs' dev curves are comparable.
        sft.add_sft_pairs(pairs, 'dev', upsample=1)
    else:
        ds = datasets.load_dataset(*RL_DATA, split='dev')
        for source, target, src_lang, tgt_lang in zip(
                ds['source_text'], ds['target_text'], ds['source_language'],
                ds['target_language']):
            if target:
                pairs.add(src_lang, tgt_lang, source, target, ORIGIN)
        # Not a warning that can be designed away: this split IS salt-69, and
        # it is used anyway because for several of these languages nothing
        # else reliable exists. Say what it costs, every run, in the log.
        log.warning(
            '--dev_source rl: every one of these dev rows is in salt-69, so '
            'the best checkpoint is selected ON the final benchmark and the '
            'salt-69 score below is optimistically biased. Chosen for '
            'coverage: 69 of 74 salt-69 target languages at a flat 10 rows '
            'per direction, against 65 at 3-8 rows for --dev_source sft. '
            'Report it as selected-on, or re-evaluate the saved checkpoint '
            'on data it never chose.')
        eval_texts = set()  # the leakage filter would drop every row
    return finalise_pairs(pairs, tokenizer, lang_ids, eval_texts, args, 'dev',
                          limit=args.dev_limit, seed=args.seed)


def prompt_blocks(directions, block_size, rng, group_by_direction=False):
    '''Row indices in blocks, one block per training step.

    Mixed by default. A long-tail language has only tens of prompts, so one
    direction per block concentrates its entire contribution into a handful of
    isolated steps -- Luhya's 87 round-trip rows are 5 blocks of 16 among 512
    steps, each landing at whatever the schedule's learning rate happens to be
    -- which is close to the worst delivery schedule available for it. Mixing
    spreads those rows over tens of steps, averaged each time with other
    languages. generate_rollouts takes a per-row target-language prefix, so
    mixed blocks need nothing else from the training loop.

    What mixing costs: padding, since source lengths and scripts no longer
    travel together, and the meaning of --scale_rewards batch, whose
    denominator becomes the reward spread *across* the step's languages (chrF
    ~0.6 for Afrikaans against ~0.2 for the weak tail) rather than within one.
    Group baselines are unaffected either way, a GRPO group being one prompt.

    group_by_direction restores one direction per block. Blocks short of
    block_size are then pooled and re-split rather than dropped, so every row
    is used and only the pooled tail mixes directions.
    '''
    if not group_by_direction:
        order = list(range(len(directions)))
        rng.shuffle(order)
        return [order[start:start + block_size]
                for start in range(0, len(order), block_size)]
    by_direction = collections.defaultdict(list)
    for i, direction in enumerate(directions):
        by_direction[direction].append(i)
    blocks, remainder = [], []
    for _, idx in sorted(by_direction.items()):
        rng.shuffle(idx)
        for start in range(0, len(idx), block_size):
            block = idx[start:start + block_size]
            (blocks if len(block) == block_size else remainder).append(block)
    tail = [i for block in remainder for i in block]
    rng.shuffle(tail)
    blocks += [tail[s:s + block_size] for s in range(0, len(tail), block_size)]
    rng.shuffle(blocks)
    return blocks


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

def generate_rollouts(model, tokenizer, lang_ids, sources, src_langs,
                      tgt_langs, args):
    '''Sample args.num_generations translations for each source text.

    Returns (encoder_input_ids, encoder_attention_mask, sequences, texts).
    `sequences` is (N*G, T), each row [decoder_start, target_language,
    tokens..., eos?] padded on the right, ordered prompt-major: row i*G+g is
    sample g of prompt i. The encoder inputs are returned so the gradient
    forward tokenises identically to generation rather than re-deriving it.

    This duplicates ~8 lines of the SFT script's translate(): the
    language-token placement and the length cap. Calling translate() instead
    would mean returning token ids and group structure from a function whose
    contract is "texts in, texts out", used on three other paths.
    '''
    device = model.device
    enc = tokenizer(list(sources), return_tensors='pt', padding=True,
                    truncation=True, max_length=args.max_length).to(device)
    # NLLB puts the language token first.
    enc['input_ids'][:, 0] = torch.tensor(
        [lang_ids[s] for s in src_langs], device=device)

    start_id = model.config.decoder_start_token_id
    if len(set(tgt_langs)) == 1:
        # forced_bos_token_id consumes one of max_new_tokens; the
        # decoder_input_ids prefix below does not, so the budget is corrected
        # to keep the number of *sampled* tokens the same either way.
        target = {'forced_bos_token_id': lang_ids[tgt_langs[0]]}
        forced_prefix_cost = 1
    else:
        target = {'decoder_input_ids': torch.tensor(
            [[start_id, lang_ids[t]] for t in tgt_langs], device=device)}
        forced_prefix_cost = 0

    new_tokens = args.rollout_max_new_tokens
    if args.rollout_length_cap:
        # A faithful translation is at most a few times its source; past that
        # the output is degenerate. Capping bounds the worst-case step time
        # without hiding the failure -- a clipped loop still reaches the
        # rewards as a long repetitive string and is punished by them.
        new_tokens = min(new_tokens, int(
            args.rollout_length_cap * enc['input_ids'].shape[1]) + 10)

    sampling = {'do_sample': True, 'temperature': args.temperature,
                'top_p': args.top_p, 'num_beams': 1}
    if args.top_k:
        sampling['top_k'] = args.top_k

    was_training = model.training
    model.eval()  # also stops transformers warning about use_cache
    with torch.no_grad():
        sequences = model.generate(
            **enc, **target, **sampling,
            num_return_sequences=args.num_generations,
            max_new_tokens=new_tokens + forced_prefix_cost,
            max_length=None, use_cache=True)
    if was_training:
        model.train()
    texts = [t.strip() for t in
             tokenizer.batch_decode(sequences, skip_special_tokens=True)]
    return enc['input_ids'], enc['attention_mask'], sequences, texts


def back_translate(model, tokenizer, lang_ids, texts, source_languages,
                   args):
    '''Translate round-trip rollouts back to English for scoring.

    Greedy (num_beams=1, no sampling): this is a measurement, not a rollout.
    Sampling here would add variance to every round-trip reward that has
    nothing to do with the rollout being scored, and within a group that
    variance does not cancel -- it is exactly what the advantage reads.

    `model` is the frozen back-translator, never the policy being trained;
    see backtranslator() for why. An empty rollout is not sent to the model
    at all: generate would return something for it, and whatever that was
    would be scored as if the policy had produced a translation.
    '''
    out = [''] * len(texts)
    index = [i for i, text in enumerate(texts) if text.strip()]
    if not index:
        return out
    translated = sft.translate(
        model, tokenizer, lang_ids, [texts[i] for i in index],
        [source_languages[i] for i in index], 'eng',
        batch_size=args.backtranslate_batch_size, num_beams=1,
        max_new_tokens=args.rollout_max_new_tokens,
        max_length=args.max_length, length_cap=args.rollout_length_cap)
    for i, text in zip(index, translated):
        out[i] = text
    return out


def completion_mask(sequences, pad_id):
    '''Which positions of `sequences[:, 1:]` the policy actually chose.

    Teacher forcing shifts by one, so position t of the labels is predicted
    from position t of the decoder inputs. Two kinds of position are excluded:
    padding after the end of a sequence, and the target-language token at
    index 0, which was forced rather than sampled and carries no gradient we
    want. EOS is deliberately kept -- emitting it is a choice worth rewarding.
    '''
    labels = sequences[:, 1:]
    mask = labels != pad_id
    mask[:, 0] = False
    return mask


def rollout_logprobs(model, enc_ids, enc_mask, sequences, pad_id):
    '''Log-probability of each sampled token, shape (B, T-1).'''
    decoder_input_ids = sequences[:, :-1]
    outputs = model(input_ids=enc_ids, attention_mask=enc_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=(decoder_input_ids != pad_id))
    logits = outputs.logits
    labels = sequences[:, 1:]
    # cross_entropy rather than an explicit log_softmax: the latter would
    # materialise a (B, T, 256k) float32 tensor, ~2x the bf16 logits that are
    # already the largest thing in the step. The (B, T) result is cast to
    # float32 before it is summed, which is where bf16 precision would bite.
    return -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
        reduction='none').view(labels.shape).float()


def group_advantages(rewards, num_generations, scale):
    '''Group-relative advantages for a step's rewards, shape (N*G,).

    The prompt's group mean is the baseline, which is what makes this work
    without a value network -- and it removes any per-language offset in the
    reward, since a group is one prompt.

    scale:
      none  -- advantage = reward - group_mean (Dr. GRPO). No difficulty bias,
               but small absolute advantages, so it wants a higher lr.
      batch -- divide by the spread over the whole step. A low-spread group
               gets a correspondingly small advantage instead of being
               amplified to unit magnitude. With mixed steps
               (the prompt_blocks default) that denominator is the spread
               *across* the step's languages, so a hard language's
               within-group gaps are divided by a figure set largely by the
               easy languages sharing the step. --block_directions keeps it
               within one language; group is immune to it.
      group -- divide by each group's own std (unit variance per group). The
               default, and the one that pairs with mixed steps: the
               denominator is within-group, so it cannot be set by the other
               languages sharing the step. The cost is that it amplifies the
               tiny gaps in a group where every rollout is equally bad to
               full strength.
    '''
    grouped = rewards.reshape(-1, num_generations)
    advantages = grouped - grouped.mean(axis=1, keepdims=True)
    if scale == 'group':
        advantages = advantages / (grouped.std(axis=1, keepdims=True) + 1e-4)
    elif scale == 'batch':
        advantages = advantages / (rewards.std() + 1e-4)
    return advantages.reshape(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def backtranslator(model, tokenizer, lang_ids, args, device):
    '''Build the callable that back-translates round-trip rollouts, or None
    if nothing needs one.

    A frozen copy of the starting checkpoint, not the policy. Two reasons,
    and the second is the one that matters. The reward stays stationary, so a
    round-trip score means the same thing at step 300 as at step 1 and the
    curve is readable. And the policy cannot co-adapt with it: with shared
    weights, "translate to Dagaare" and "translate back to English" are the
    same parameters, and a private encoding that round-trips perfectly while
    being no language at all scores 1.0 on every round-trip row. That is the
    classic degenerate solution to a round-trip objective, and freezing the
    back-translator removes it rather than penalising it.

    Under --lora_r the policy is the same base weights plus an adapter, so
    the frozen model is the policy with the adapter switched off rather than a
    second copy -- loading one would defeat the point of using LoRA to fit a
    larger model on a smaller card. The adapter goes off for the duration of
    one back-translation call and back on immediately, which is why this
    returns a callable rather than a model: holding disable_adapter() open
    across the step would also disable it for the policy gradient. Untested,
    as the rest of the LoRA path is -- peft is not installed in /venv/main.
    '''
    if not args.roundtrip:
        return None
    if args.lora_r:
        def with_adapter_off(texts, source_languages):
            with model.disable_adapter():
                return back_translate(model, tokenizer, lang_ids, texts,
                                      source_languages, args)
        return with_adapter_off
    log.info('Loading a frozen back-translator from %s (bf16, no grad). '
             'It is never updated, so a round-trip reward means the same '
             'thing at the last step as at the first', args.base_model)
    frozen = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16).to(device).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return lambda texts, source_languages: back_translate(
        frozen, tokenizer, lang_ids, texts, source_languages, args)


def apply_lora(model, args):
    '''Wrap the model in a LoRA adapter (optional; needs peft installed).

    Full fine-tuning is the default, matching the supervised script, but it
    holds gradients and optimizer state for every parameter. LoRA is the way
    to run the 3.3B on a smaller card, and is what published GRPO runs on
    NLLB used. M2M100 has no entry in peft's target-module table, so the
    modules are named explicitly.
    '''
    try:
        import peft
    except ImportError:
        raise SystemExit(
            '--lora_r needs peft: pip install peft (or use --lora_r 0 for '
            'full fine-tuning).')
    config = peft.LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, bias='none',
        task_type=peft.TaskType.SEQ_2_SEQ_LM,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj',
                        'fc1', 'fc2'])
    model = peft.get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info('LoRA r=%d on q,k,v,out,fc1,fc2: %s trainable of %s (%.2f%%)',
             args.lora_r, f'{trainable:,}', f'{total:,}',
             100 * trainable / total)
    return model


def write_language_codes(path, args):
    '''The code map the SFT script writes beside every final model.'''
    codes = dict(sft.LANGS)
    if args.src_lang_dropout:
        codes[sft.UNKNOWN_SOURCE] = constants.NLLB_69_UNKNOWN_SOURCE_CODE
    with open(Path(path) / 'nllb_language_codes.json', 'w') as f:
        json.dump(codes, f, indent=2)


def save_checkpoint(model, tokenizer, path, args):
    '''Save a checkpoint mid-run, in the layout the SFT script's `eval`
    expects: a full model, or a LoRA adapter.

    Under LoRA this writes the adapter alone, which is small and lossless.
    Merging here instead would be wrong twice over: merge_and_unload returns
    the base model and leaves the wrapper unusable for the rest of training,
    and merge/unmerge round trips lose precision in bf16. The merge happens
    once, in merge_lora(), after training.'''
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    write_language_codes(path, args)
    log.info('Saved %s to %s', 'adapter' if args.lora_r else 'model', path)


def merge_lora(adapter_path, destination, tokenizer, args):
    '''Load the base model, apply the saved adapter and merge, so the result
    loads with AutoModelForSeq2SeqLM like every other checkpoint in this
    family. Done on a freshly loaded base rather than the training copy, whose
    weights the optimizer has been updating.'''
    import peft
    base = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16)
    merged = peft.PeftModel.from_pretrained(base, str(adapter_path))
    merged = merged.merge_and_unload()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(destination))
    tokenizer.save_pretrained(str(destination))
    write_language_codes(destination, args)
    del base, merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info('Merged %s into %s', adapter_path, destination)


def log_metrics(metrics, step):
    if sft.mlflow_enabled():
        import mlflow
        mlflow.log_metrics(metrics, step=step)


def log_rollout_samples(sources, texts, references, total, components,
                        num_generations, step, back_translations=None):
    '''Show the group with the widest reward spread: its best and worst
    rollout, with the reward breakdown. One group per call -- enough to see
    what the reward is actually paying for, cheap enough to leave on.

    On a round-trip group the back-translation is shown too: it is what chrF
    actually scored, so without it the numbers cannot be read against the
    text.'''
    grouped = total.reshape(-1, num_generations)
    i = int(np.argmax(grouped.max(axis=1) - grouped.min(axis=1)))
    group = total[i * num_generations:(i + 1) * num_generations]
    lines = [f'step {step} | source: {sources[i][:160]}',
             f'  reference: {references[i * num_generations][:160]}']
    for label, g in (('best ', int(np.argmax(group))),
                     ('worst', int(np.argmin(group)))):
        j = i * num_generations + g
        parts = ' '.join(f'{k}={v[j]:+.3f}' for k, v in components.items())
        lines.append(f'  {label} r={total[j]:+.3f} ({parts}): '
                     f'{texts[j][:160]}')
        if back_translations and back_translations[j]:
            lines.append(f'    back-translated: {back_translations[j][:160]}')
    log.info('\n'.join(lines))


def evaluate_dev(model, tokenizer, lang_ids, dev, args):
    '''Greedy dev chrF, and the same without the source language when the
    model was trained to make it optional.'''
    was_training = model.training
    model.eval()
    metrics = {}
    with torch.no_grad():
        for prefix, unknown_source in (('', False), ('nosrc_', True)):
            if unknown_source and not args.src_lang_dropout:
                continue
            scores = sft.dev_chrf(
                model, tokenizer, lang_ids, dev,
                batch_size=args.dev_chrf_batch_size,
                max_length=args.max_length, unknown_source=unknown_source)
            metrics.update({f'eval_{prefix}{k}': v for k, v in scores.items()})
    if was_training:
        model.train()
    return metrics


def train(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = random.Random(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    lang_ids = constants.nllb_69_language_token_ids(tokenizer)
    eval_texts = sft.salt69_texts()
    train_rows = load_rl_prompts(args, tokenizer, lang_ids, eval_texts)
    dev = load_dev_pairs(args, tokenizer, lang_ids, eval_texts)

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16).to(device)
    if args.lora_r:
        model = apply_lora(model, args)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()

    import torchao.optim
    # Pure-bf16 weights, as in the supervised script: stochastic rounding
    # stops small updates being rounded away, which matters more here than
    # there -- a policy-gradient update is smaller and noisier than an MLE one.
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torchao.optim.AdamW8bit(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay,
        bf16_stochastic_round=True)

    back_translate_rollouts = backtranslator(
        model, tokenizer, lang_ids, args, device)
    if back_translate_rollouts and args.langid_weight:
        # Fail here rather than 50 steps in, and put it on the policy's
        # device: it is 62MB against the policy's 2.7GB.
        load_language_classifier(device)

    blocks = prompt_blocks(
        list(zip(train_rows['source.language'], train_rows['target.language'])),
        args.prompts_per_step, rng, args.block_directions)
    if not blocks:
        raise SystemExit('No training pairs survived filtering.')
    steps_per_epoch = len(blocks)
    total_steps = (args.max_steps if args.max_steps > 0
                   else int(args.num_train_epochs * steps_per_epoch))
    if total_steps < 1:
        raise SystemExit(
            f'{total_steps} steps to run: --num_train_epochs '
            f'{args.num_train_epochs} over {steps_per_epoch} blocks rounds '
            'down to nothing. Lower --prompts_per_step or raise the epochs.')
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, args.warmup_steps, total_steps)
    log.info('%d prompts in %d blocks of %d; %d steps (%.2f epochs), '
             '%d rollouts per step', len(train_rows['source']),
             steps_per_epoch, args.prompts_per_step, total_steps,
             total_steps / steps_per_epoch,
             args.prompts_per_step * args.num_generations)

    if sft.mlflow_enabled():
        import mlflow
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=out.name)
        mlflow.log_params({k: v for k, v in vars(args).items()
                           if v is not None and not isinstance(v, list)})
        mlflow.log_param('total_steps', total_steps)

    pad_id = model.config.pad_token_id
    eos_id = model.config.eos_token_id
    best = {'metric': -math.inf, 'step': -1}
    evals_without_improvement = 0
    history = collections.deque(maxlen=args.logging_steps)
    step = 0
    stop = False

    while step < total_steps and not stop:
        rng.shuffle(blocks)
        for block in blocks:
            if step >= total_steps or stop:
                break
            t0 = time.time()
            sources = [train_rows['source'][i] for i in block]
            src_langs = [train_rows['source.language'][i] for i in block]
            tgt_langs = [train_rows['target.language'][i] for i in block]
            if args.src_lang_dropout:
                # Decided per prompt, not per rollout: a group's rollouts must
                # share one prompt or their advantages compare different
                # questions. The rate matches training, so RL does not pull
                # the model off the unknown-source mode it was taught.
                src_langs = [sft.UNKNOWN_SOURCE
                             if rng.random() < args.src_lang_dropout else s
                             for s in src_langs]
            # Per rollout, not per prompt: every reward component is
            # scored per rollout, and a block can hold both kinds of row
            # whenever a language has labelled and round-trip rows in the
            # same direction.
            def per_rollout(values):
                return [v for v in values for _ in range(args.num_generations)]

            references = per_rollout(train_rows['target'][i] for i in block)
            roundtrip = per_rollout(
                train_rows['origin'][i] == ORIGIN_ROUNDTRIP for i in block)
            # The source text itself, for the copy penalty to compare the
            # rollout against. Every round-trip row is eng->X, so on the rows
            # the penalty applies to this is the English side. src_langs may
            # have been blanked by --src_lang_dropout; the text has not.
            rollout_sources = per_rollout(sources)
            rollout_tgt_langs = per_rollout(tgt_langs)

            enc_ids, enc_mask, sequences, texts = generate_rollouts(
                model, tokenizer, lang_ids, sources, src_langs, tgt_langs,
                args)
            t_gen = time.time() - t0

            back_translations = None
            t_back = 0.0
            if any(roundtrip):
                t_back = time.time()
                index = [i for i, r in enumerate(roundtrip) if r]
                translated = back_translate_rollouts(
                    [texts[i] for i in index],
                    [rollout_tgt_langs[i] for i in index])
                back_translations = [''] * len(texts)
                for i, text in zip(index, translated):
                    back_translations[i] = text
                t_back = time.time() - t_back

            total, components = score_rollouts(
                texts, references, args, sources=rollout_sources,
                target_languages=rollout_tgt_langs, roundtrip=roundtrip,
                back_translations=back_translations, device=device)
            advantages = group_advantages(
                total, args.num_generations, args.scale_rewards)
            t_reward = time.time() - t0 - t_gen - t_back

            mask = completion_mask(sequences, pad_id)
            n_rollouts = sequences.shape[0]
            # The normalising denominator is the whole step's, not the
            # micro-batch's, so micro-batching is exactly gradient
            # accumulation and changing it does not change the update.
            denominator = (float(mask.sum()) if args.loss_normalization == 'token'
                           else float(n_rollouts))
            advantage_t = torch.tensor(advantages, dtype=torch.float32,
                                       device=device)
            prompt_index = torch.arange(
                len(block), device=device).repeat_interleave(
                    args.num_generations)

            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for start in range(0, n_rollouts, args.rollout_micro_batch):
                rows = slice(start, start + args.rollout_micro_batch)
                prompts = prompt_index[rows]
                logps = rollout_logprobs(
                    model, enc_ids[prompts], enc_mask[prompts],
                    sequences[rows], pad_id)
                micro_mask = mask[rows]
                if args.loss_normalization == 'token':
                    weighted = (advantage_t[rows].unsqueeze(1) * logps
                                * micro_mask).sum()
                else:
                    per_sequence = ((logps * micro_mask).sum(1)
                                    / micro_mask.sum(1).clamp(min=1))
                    weighted = (advantage_t[rows] * per_sequence).sum()
                loss = -weighted / denominator
                loss.backward()
                loss_total += float(loss.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, args.max_grad_norm)
            # Read before stepping: afterwards it is the next step's rate.
            learning_rate = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step()
            step += 1

            grouped = total.reshape(-1, args.num_generations)
            lengths = mask.sum(1).float()
            terminated = (sequences[:, 2:] == eos_id).any(1).float()
            metrics = {
                'train_loss': loss_total,
                'train_grad_norm': float(grad_norm),
                'train_lr': learning_rate,
                'reward_total': float(total.mean()),
                'reward_spread': float((grouped.max(axis=1)
                                        - grouped.min(axis=1)).mean()),
                # The metric to watch. As it approaches 1.0 every group's
                # rollouts agree, so every advantage is zero and the run has
                # stopped learning whatever the loss says.
                'rollouts_zero_variance_groups': float(
                    (grouped.std(axis=1) < 1e-6).mean()),
                'rollouts_mean_tokens': float(lengths.mean()),
                'rollouts_max_tokens': float(lengths.max()),
                'rollouts_eos_rate': float(terminated.mean()),
                'rollouts_repetition_rate': float(
                    (components['repetition'] < 0).mean()),
                'rollouts_roundtrip_rate': float(np.mean(roundtrip)),
                'time_generation': t_gen,
                'time_backtranslation': t_back,
                'time_reward': t_reward,
                'time_step': time.time() - t0,
            }
            # Branch-split chrF, because the two measure different things: a
            # labelled row's chrF is the rollout against a target-language
            # reference, a round-trip row's is its back-translation against
            # English. Averaging them into one number would be meaningless.
            # Only present on steps that have that kind of row, which is why
            # the running mean below tolerates a missing key.
            mask_rt = np.asarray(roundtrip, dtype=bool)
            # A group is one prompt and a prompt's branch is fixed, so every
            # rollout of a group is on the same branch and this is exact.
            group_is_rt = mask_rt.reshape(-1, args.num_generations)[:, 0]
            flat = grouped.std(axis=1) < 1e-6
            # Worth splitting, not just aggregating. Three of the
            # round-trip penalties can each reach their floor independently,
            # so the clamp binds far more often on that branch -- and when it
            # binds for every rollout of a group, the only signal left is the
            # back-translation's chrF. If zero_variance_roundtrip climbs
            # towards 1.0 while the labelled one stays low, the branch has
            # stopped teaching anything and --max_penalty is the lever, not
            # --temperature.
            #
            # nan, not a skipped key, when a step has none of a branch: the
            # keys then stay the same on every step, so the MLflow chart is
            # one series with gaps rather than a series that appears and
            # disappears, and the running mean below ignores the gaps.
            for name, rollouts, groups in (
                    ('roundtrip', mask_rt, group_is_rt),
                    ('labelled', ~mask_rt, ~group_is_rt)):
                metrics[f'reward_chrf_{name}'] = (
                    float(components['chrf'][rollouts].mean())
                    if rollouts.any() else math.nan)
                metrics[f'rollouts_zero_variance_{name}'] = (
                    float(flat[groups].mean())
                    if groups.any() else math.nan)
            metrics.update({f'reward_{k}': float(v.mean())
                            for k, v in components.items()})
            history.append(metrics)

            if step % args.logging_steps == 0 or step == 1:
                # nanmean, because the per-branch keys are nan on a step
                # that had none of that branch; a window with none at all
                # stays nan rather than warning.
                def window_mean(key):
                    values = [h[key] for h in history if key in h]
                    values = [v for v in values if not math.isnan(v)]
                    return float(np.mean(values)) if values else math.nan

                mean = {k: window_mean(k) for k in metrics}
                log.info(
                    'step %d/%d | reward %.3f (chrf %.3f pen %.3f) | '
                    'spread %.3f zero-var %.2f | %.0f tok eos %.2f rep %.2f '
                    'rt %.2f zv-rt %.2f | loss %.4f gn %.2f lr %.2g | '
                    '%.1fs (gen %.1f bt %.1f)',
                    step, total_steps, mean['reward_total'],
                    mean['reward_chrf'], mean['reward_penalty'],
                    mean['reward_spread'],
                    mean['rollouts_zero_variance_groups'],
                    mean['rollouts_mean_tokens'], mean['rollouts_eos_rate'],
                    mean['rollouts_repetition_rate'],
                    mean['rollouts_roundtrip_rate'],
                    mean['rollouts_zero_variance_roundtrip'],
                    mean['train_loss'],
                    mean['train_grad_norm'], mean['train_lr'],
                    mean['time_step'], mean['time_generation'],
                    mean['time_backtranslation'])
                log_metrics(mean, step)
            if args.log_samples_steps and step % args.log_samples_steps == 0:
                log_rollout_samples(sources, texts, references, total,
                                    components, args.num_generations, step,
                                    back_translations)

            if step % args.eval_steps == 0 or step == total_steps:
                scores = evaluate_dev(model, tokenizer, lang_ids, dev, args)
                log.info('step %d eval: %s', step, scores)
                log_metrics(scores, step)
                metric = scores[f'eval_{args.best_metric}']
                if metric > best['metric']:
                    best.update(metric=metric, step=step)
                    evals_without_improvement = 0
                    save_checkpoint(model, tokenizer, out / 'best', args)
                    log.info('New best %s=%.4f at step %d',
                             args.best_metric, metric, step)
                else:
                    evals_without_improvement += 1
                    log.info('No improvement on %s (best %.4f at step %d); '
                             '%d/%d evaluations without one',
                             args.best_metric, best['metric'], best['step'],
                             evals_without_improvement,
                             args.early_stopping_patience)
                    if (args.early_stopping_patience
                            and evals_without_improvement
                            >= args.early_stopping_patience):
                        log.info('Early stopping: the best checkpoint is '
                                 'step %d.', best['step'])
                        stop = True

    if torch.cuda.is_available():
        log.info('Peak GPU memory: %.1f GB',
                 torch.cuda.max_memory_allocated() / 1e9)

    # The best dev checkpoint is the model, not the last one: published GRPO
    # runs on NLLB report collapse late in training as the reward spread
    # within groups shrinks, so "last" is not a safe default here.
    final = out / 'final'
    source = out / 'best' if best['step'] > 0 and (out / 'best').exists() else None
    if source is None:
        log.warning('No evaluation improved on the starting point, so there '
                    'is no best checkpoint; saving the last one instead.')
        source = out / 'last'
        save_checkpoint(model, tokenizer, source, args)
    else:
        log.info('Final model is the best checkpoint (step %d, %s=%.4f)',
                 best['step'], args.best_metric, best['metric'])
    if final.exists():
        shutil.rmtree(final)
    if args.lora_r:
        merge_lora(source, final, tokenizer, args)
    else:
        shutil.copytree(source, final)
        log.info('Final model at %s', final)
    if sft.mlflow_enabled():
        import mlflow
        mlflow.log_metric(f'best_eval_{args.best_metric}', best['metric'])
        mlflow.log_param('best_step', best['step'])
    if args.hub_model_id:
        log.info('Pushing %s to %s', final, args.hub_model_id)
        pushed = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            str(final), dtype=torch.bfloat16)
        pushed.push_to_hub(args.hub_model_id, private=True)
        tokenizer.push_to_hub(args.hub_model_id, private=True)
    return str(final)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('command', choices=['train', 'eval', 'all'])
    p.add_argument('--output_dir', required=True)
    p.add_argument('--base_model', required=True,
                   help='Model to continue with RL: a checkpoint from '
                        'translation_nllb_training_69_languages.py, e.g. '
                        'OUTPUT_DIR/final. Starting from stock NLLB-200 '
                        'would work but wastes RL on what supervised '
                        'training does better')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mlflow', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Log to MLflow (needs MLFLOW_TRACKING_URI etc.); '
                        '--no-mlflow runs without logging')
    p.add_argument('--max_length', type=int, default=512,
                   help='Max source/target tokens; longer pairs are dropped')
    p.add_argument('--max_source_tokens', type=int, default=256,
                   help='Also drop TRAINING prompts whose source is longer '
                        'than this (the dev set is never capped). Tighter '
                        'than --max_length because rollout cost scales with '
                        'the longest source in the batch, and this corpus '
                        'mixes sentences with paragraphs: over the 7,535 '
                        'labelled train rows the source is 88 tokens at p50 '
                        'but 450 at p90 and 1,233 at most. This cap keeps '
                        '5,098 rows (68%%) at a mean of 72 tokens, against '
                        '7,005 (93%%) at 512 and a mean of 147 -- so it '
                        'roughly halves the cost per rollout for a third of '
                        'the prompts. It also matches what salt-69 measures, '
                        'which is sentence-level translation. 0 keeps '
                        'everything up to --max_length')
    p.add_argument('--src_lang_dropout', type=float, default=0.25,
                   help='Fraction of prompts given the unknown-source token '
                        'instead of their source language. Match the value '
                        'the base model was trained with, or RL pulls it off '
                        'that mode. 0 = always give the source language')

    g = p.add_argument_group('rollouts')
    g.add_argument('--num_generations', type=int, default=12,
                   help='Rollouts per prompt (the GRPO group). Below ~4 the '
                        'group mean is too noisy to be a baseline. Raising 8 '
                        '-> 12 costs 50%% more generation per step and cuts '
                        'the rate of groups with no usable rollout by '
                        '60-80%% in the mid-tail languages (--floor_reward)')
    g.add_argument('--block_directions',
                   action=argparse.BooleanOptionalAction, default=False,
                   help='Put one translation direction in each step instead '
                        'of mixing them. Better padding, and it keeps the '
                        '--scale_rewards batch denominator within one '
                        'language, but it delivers a long-tail language\'s '
                        'whole contribution in a handful of isolated steps; '
                        'see prompt_blocks')
    g.add_argument('--prompts_per_step', type=int, default=16,
                   help='Prompts per optimizer step; rollouts per step is '
                        'this x --num_generations')
    g.add_argument('--temperature', type=float, default=1.1,
                   help='Rollout sampling temperature. A supervised NMT model '
                        'is near-deterministic, so too low a value makes '
                        'every rollout in a group identical and the advantage '
                        'zero -- watch rollouts_zero_variance_groups')
    g.add_argument('--top_p', type=float, default=1.0)
    g.add_argument('--top_k', type=int, default=0,
                   help='0 disables top-k')
    g.add_argument('--rollout_max_new_tokens', type=int, default=256)
    g.add_argument('--rollout_length_cap', type=float, default=3.0,
                   help='Cap generation at this multiple of the batch\'s '
                        'longest source, + 10 tokens. Bounds a '
                        'non-terminating rollout\'s cost; it is still scored '
                        '(and punished) as the long repetitive string it is. '
                        '0 disables')

    g = p.add_argument_group('reward')
    g.add_argument('--chrf_weight', type=float, default=1.0)
    g.add_argument('--repetition_weight', type=float, default=1.0)
    g.add_argument('--length_weight', type=float, default=1.0)
    g.add_argument('--copy_weight', type=float, default=1.0,
                   help='Weight of the untranslated-source penalty. '
                        'Round-trip rows only, where echoing the English '
                        'source back-translates perfectly and would '
                        'otherwise score ~1.0 for translating nothing')
    g.add_argument('--langid_weight', type=float, default=1.0,
                   help='Weight of the wrong-language penalty. Round-trip '
                        'rows only. 0 disables it, and also stops dropping '
                        'rows whose language the classifier cannot detect -- '
                        'see load_rl_prompts for what that costs')
    g.add_argument('--langid_top_k', type=int, default=LANGID_TOP_K,
                   help='Charge the flat wrong-language penalty only when '
                        'the target language falls outside the classifier\'s '
                        'top k labels, so mutually confusable neighbours go '
                        'unpenalised. 1 charges whenever its single best '
                        'label is not the target. See language_scores')
    g.add_argument('--floor_reward', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Clip the total reward at 0, so a rollout whose '
                        'penalties outweigh its chrF scores exactly 0 rather '
                        'than going negative. --no-floor_reward restores the '
                        'unbounded total that runs 1 and 2 used; see '
                        'MAX_COMBINED_PENALTY')
    g.add_argument('--max_penalty', type=float, default=MAX_COMBINED_PENALTY,
                   help='Cap, as a magnitude, on the summed repetition and '
                        'length penalties, so a rollout tripping both is '
                        'charged once; see MAX_COMBINED_PENALTY. 2.0 '
                        'recovers the uncapped sum')
    g.add_argument('--scale_rewards', choices=['none', 'batch', 'group'],
                   default='group',
                   help='How group advantages are normalised; see '
                        'group_advantages')

    g = p.add_argument_group(
        'round-trip rows (no reference translation)')
    g.add_argument('--roundtrip', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Train on the 3,199 rows with no reference '
                        'translation by translating each rollout back to '
                        'English and scoring that. --no-roundtrip drops them '
                        'and trains on labelled pairs only, which cannot '
                        'reach 7 of the languages at all')
    g.add_argument('--roundtrip_exclude_languages', nargs='+',
                   default=['kln'],
                   help='Target languages to keep out of the round-trip '
                        'branch. Default kln: its NLLB token is cat_Latn and '
                        'the classifier has no Kalenjin label, so it is the '
                        'one language that is both reward-inverted and '
                        'unguarded')
    g.add_argument('--backtranslate_batch_size', type=int, default=32,
                   help='Back-translation runs under no_grad, so this can be '
                        'well above --rollout_micro_batch')

    g = p.add_argument_group('optimisation')
    g.add_argument('--learning_rate', type=float, default=2e-6,
                   help='Much lower than supervised training: the gradient is '
                        'a policy gradient on the model\'s own samples, and '
                        'this is a short run on top of a trained model')
    g.add_argument('--weight_decay', type=float, default=0.0)
    g.add_argument('--warmup_steps', type=int, default=20)
    g.add_argument('--num_train_epochs', type=float, default=1.0)
    g.add_argument('--max_steps', type=int, default=-1,
                   help='Overrides --num_train_epochs when > 0')
    g.add_argument('--max_grad_norm', type=float, default=1.0)
    g.add_argument('--rollout_micro_batch', type=int, default=8,
                   help='Rollouts per forward/backward. Pure memory knob -- '
                        'the loss denominator is the whole step\'s, so this '
                        'does not change the update. The logits are '
                        '(batch x length x 256k vocab), ~0.4GB in bf16 at 8 x '
                        '100 tokens, and the backward holds a gradient of the '
                        'same size')
    g.add_argument('--loss_normalization', choices=['sequence', 'token'],
                   default='sequence',
                   help='sequence: mean log-prob per rollout, then mean over '
                        'rollouts, so every rollout counts the same whatever '
                        'its length. token: sum over all sampled tokens in '
                        'the step, which weights long rollouts more')
    g.add_argument('--gradient_checkpointing',
                   action=argparse.BooleanOptionalAction, default=False)
    g.add_argument('--lora_r', type=int, default=0,
                   help='0 = full fine-tuning. > 0 trains a LoRA adapter '
                        '(needs peft) and merges it before saving, which is '
                        'how to fit the 3.3B on a smaller card')
    g.add_argument('--lora_alpha', type=int, default=32)
    g.add_argument('--lora_dropout', type=float, default=0.05)
    g.add_argument('--limit_rows', type=int,
                   help='Use only the first N rl_translations rows (testing)')

    g = p.add_argument_group('evaluation during training')
    g.add_argument('--dev_source', choices=['sft', 'rl'], default='sft',
                   help='Which dev set drives --best_metric and early '
                        'stopping. sft: sft_translations dev, 65 of the 74 '
                        'salt-69 target languages at 3-8 rows each, clean of '
                        'salt-69, comparable with the supervised runs. rl: '
                        'rl_translations dev, 69 of 74 at a flat 10 rows, but '
                        'drawn entirely from salt-69, so the final score is '
                        'selected-on. Every language sft lacks also has no '
                        'labelled training data, so for this reference-only '
                        'phase the coverage gap costs regression detection '
                        'rather than improvement detection')
    g.add_argument('--eval_steps', type=int, default=50)
    g.add_argument('--dev_limit', type=int,
                   help='Sample this many sft_translations dev pairs '
                        '(testing). Sampled, not truncated: dev is ordered '
                        'by language')
    g.add_argument('--dev_chrf_batch_size', type=int, default=64)
    g.add_argument('--best_metric',
                   choices=['chrf', 'chrf_to_eng', 'chrf_from_eng',
                            'chrf_non_eng', 'nosrc_chrf'],
                   default='chrf',
                   help='Dev metric for the best checkpoint and early '
                        'stopping')
    g.add_argument('--early_stopping_patience', type=int, default=5,
                   help='Stop after this many evaluations without '
                        'improvement. 0 disables. On by default: GRPO runs on '
                        'NLLB are reported to collapse late as the reward '
                        'spread within groups shrinks')
    g.add_argument('--logging_steps', type=int, default=10)
    g.add_argument('--log_samples_steps', type=int, default=50,
                   help='Log the best and worst rollout of one group every N '
                        'steps. 0 disables')
    g.add_argument('--hub_model_id', help='Push the final model here '
                                          '(private)')

    g = p.add_argument_group('salt-69 evaluation (the SFT script\'s, '
                             'unchanged)')
    g.add_argument('--model', help='Model to evaluate (default: '
                                   'OUTPUT_DIR/final)')
    g.add_argument('--languages', nargs='+',
                   help='Languages to evaluate (default: all in salt-69)')
    g.add_argument('--eval_limit', type=int,
                   help='Max salt-69 sentences per language (testing)')
    g.add_argument('--eval_batch_size', type=int, default=32)
    g.add_argument('--num_beams', type=int, default=4)
    g.add_argument('--max_new_tokens', type=int, default=256)
    g.add_argument('--run_name', help='Name for result files')
    g.add_argument('--source_modes', nargs='+', choices=['given', 'unknown'],
                   help='Evaluate with the source language given, not given, '
                        'or both. Default: both if --src_lang_dropout > 0')

    args = p.parse_args()
    if args.temperature <= 0:
        p.error('--temperature must be > 0: GRPO needs the rollouts within a '
                'group to differ, or every advantage is zero')
    if args.best_metric == 'nosrc_chrf' and not args.src_lang_dropout:
        p.error('--best_metric nosrc_chrf needs --src_lang_dropout > 0')
    if args.prompts_per_step * args.num_generations % args.rollout_micro_batch:
        log.warning('rollouts per step (%d) is not a multiple of '
                    '--rollout_micro_batch (%d); the last micro-batch of each '
                    'step is smaller, which is harmless',
                    args.prompts_per_step * args.num_generations,
                    args.rollout_micro_batch)
    if args.source_modes is None:
        args.source_modes = (['given', 'unknown'] if args.src_lang_dropout
                             else ['given'])
    args.source_modes = [m for m in ('given', 'unknown')
                         if m in args.source_modes]
    return args


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    transformers.set_seed(args.seed)
    if args.mlflow:
        sft.check_mlflow(MLFLOW_EXPERIMENT)
    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        sft.evaluate(args)
    else:
        final_dir = train(args)
        sft.evaluate(args, model_path=final_dir)
    if sft.mlflow_enabled():
        import mlflow
        mlflow.end_run()


if __name__ == '__main__':
    main()
