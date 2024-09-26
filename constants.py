SALT_LANGUAGE_NAMES = {
    'ach': 'Acholi',
    'eng': 'English',
    'ibo': 'Igbo',
    'lgg': 'Lugbara',
    'lug': 'Luganda',
    'nyn': 'Runyankole',
    'swa': 'Swahili',
    'teo': 'Ateso',
}

SALT_LANGUAGE_TOKENS_WHISPER = {
    # Exact/close mapping
    'eng': 50259,
    'swa': 50318,
    # Overwrite unused language tokens
    'ach': 50357,
    'lgg': 50356,
    'lug': 50355,
    'nyn': 50354,
    'teo': 50353,
}

SALT_LANGUAGE_TOKENS_NLLB_TRANSLATION = {
    # Exact/close mapping
    'eng': 'eng_Latn',
    'lug': 'lug_Latn',
    'ach': 'luo_Latn',
    'ibo': 'ibo_Latn',
    'swa': 'swh_Latn',
    # Overwrite unused language tokens
    'nyn': 'ace_Latn',
    'teo': 'afr_Latn',
    'lgg': 'aka_Latn',
}