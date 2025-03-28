SALT_LANGUAGE_NAMES = {
    'ach': 'Acholi',
    'eng': 'English',
    'ibo': 'Igbo',
    'lgg': 'Lugbara',
    'lug': 'Luganda',
    'nyn': 'Runyankole',
    'swa': 'Swahili',
    'teo': 'Ateso',
    'xog': 'Lusoga',
    'ttj': 'Rutooro',
    'kin': 'Kinyarwanda',
    'myx': 'Lumasaba',
    'adh': 'Jopadhola',
    'alz': 'Alur',
    'bfa': 'Bari',
    'cgg': 'Rukiga',
    'gwr': 'Lugwere',
    'ikx': 'Ik',
    'kdi': 'Kumam',
    'kdj': 'Karamojong',
    'keo': 'Kakwa',
    'koo': 'Rukonjo',
    'kpz': 'Kupsabiny',
    'laj': 'Lango',
    'led': 'Lendu',
    'lsm': 'Samia',
    'lth': 'Thur',
    'luc': 'Aringa',
    'lug': 'Luganda',
    'lzm': 'Lulubo',
    'mhi': 'Ma\'di',
    'ndp': 'Ndo',
    'pok': 'Pokot',
    'rub': 'Lugungu',
    'ruc': 'Ruruuli',
    'rwm': 'Kwamba',
    'sbx': 'Sebei',
    'soc': 'So',
    'tlj': 'Bwisi-Talinga',
    'nuj': 'Lunyole',
    'nyo': 'Runyoro',
    'luo': 'Luo',
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
    'xog': 50352,
    'ttj': 50351,
    'kin': 50350,
    'myx': 50349,
}

SALT_LANGUAGE_TOKENS_NLLB_TRANSLATION = {
    # Exact/close mapping
    'eng': 256047, # eng_Latn
    'lug': 256110, # lug_Latn
    'ach': 256111, # luo_Latn
    'ibo': 256073, # ibo_Latn
    'swa': 256168, # swh_Latn
    # Overwrite unused language tokens
    'nyn': 256002,
    'teo': 256006,
    'lgg': 256008,
    'xog': 256009,
    'ttj': 256010,
    'myx': 256011,
}
