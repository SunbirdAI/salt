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

def show_dataset(ds, N=10, rate=16_000):
    '''Show dataset inside a Jupyter notebook with embedded audio.'''
    def create_audio_player_from_array(audio_data_array):
        audio_player = Audio(data=audio_data_array['array'], rate=rate)
        return audio_player._repr_html_().replace('\n','')

    df_audio = pd.DataFrame(ds[:N])
    audio_keys = []
    for k, v in ds[0].items():
        if isinstance(v, dict) and 'array' in v:
            audio_keys.append(k)
    for k in audio_keys:
        df_audio[k] = df_audio[k].apply(create_audio_player_from_array)

    display.display(display.HTML(df_audio.to_html(escape=False)))