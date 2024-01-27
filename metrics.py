import numpy as np
import pandas as pd

def multilingual_eval(eval_preds,
                      source_language,
                      target_language,
                      metrics,
                      metric_names,
                      tokenizer,
                      log_first_N_predictions):
    '''Compute metric scores for each source and target language combination.'''
    
    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    decoded_predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    if log_first_N_predictions:
        for i in range(log_first_N_predictions):
            print(f'prediction: {decoded_predictions[i]}, '
                  f'true: {decoded_labels[i]}')

    subsets = {}
    for i in range(len(predictions)):
        language_combination = source_language[i] + '_' + target_language[i]
        if language_combination not in subsets:
            subsets[language_combination] = {'predictions': [], 'labels': []}
        subsets[language_combination]['predictions'].append(
            decoded_predictions[i])
        subsets[language_combination]['labels'].append(decoded_labels[i])
           
    result = {}
    for metric, metric_name in zip(metrics, metric_names):
        for subset in list(subsets.keys()):
            result_subset = metric.compute(
                predictions=subsets[subset]['predictions'],
                references=subsets[subset]['labels'])
            if metric_name == 'BLEU':
                # The sacrebleu and bleu implementations have different formats
                r = result_subset.get('score') or result_subset.get('bleu')
            elif metric_name == 'WER':
                r = result_subset
            else:
                raise ValueError('Only BLEU and WER metrics currently '
                                 'supported.')
            result[f'{metric_name}_{subset}'] = r 
        
    result[f'{metric_name}_mean'] = np.mean(
        [result[f'{metric_name}_{subset}'] for subset in list(subsets.keys())]
    )

    result = {k: round(v, 3) for k, v in result.items()}
    return result

def multilingual_eval_fn(eval_dataset,
                         metrics,
                         tokenizer,
                         log_first_N_predictions=3):
    '''Return a function with the signature `eval_fn(preds)`.'''
   
    df = pd.DataFrame(eval_dataset)
    source_language = list(df['source.language'])
    target_language = list(df['target.language'])
    
    metric_names = []
    for m in metrics:
        if m.name == 'sacrebleu':
            metric_names.append('BLEU')
        else:
            metric_names.append(m.name)
        
    return functools.partial(multilingual_eval,
                             source_language,
                             target_language,
                             metrics,
                             metric_names,
                             tokenizer,
                             log_first_N_predictions=log_first_N_predictions)