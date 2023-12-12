import functools

def single_batch_entry(func):
    """Split a batch into individual items, process and then recombine."""
    @functools.wraps(func)
    def single_batch_entry(r, src_or_tgt, **kwargs):
        keys = r.keys()        
        result = {k: [] for k in keys}
        for i in range(len(r['source'])):
            single_entry = {k: r[k][i] for k in keys}
            single_result = func(single_entry, src_or_tgt, **kwargs)
            for k in keys:
                result[k].append(single_result[k])
        return result
    return single_batch_entry
