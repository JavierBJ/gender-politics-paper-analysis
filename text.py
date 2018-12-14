"""gender-politics-paper-analysis: text.py

This module is in charge of the pre-processing of tweets. It uses FreeLing 4.0, NLTK and an emoji library for that task,
so it is basically a handler for these tools. It also contains hand-crafted filters that keep some tokens out of the
dataset.

If you want to use another pre-processing strategy or just other libraries, you can ignore this module and change
the call to preprocess() in the analyze.py module for your own code.

Author: Javier Beltran Jorba
Institut Barcelona d'Estudis Internacionals
Universitat Pompeu Fabra
"""


import pyfreeling
import configparser
from nltk.corpus import stopwords


def setup_freeling():
    """
    Loads FreeLing with the settings for Spanish. It's called inside preprocess(). Paths with the Spanish settings
    are read from the file config.ini. If your installation of FreeLing differs from the typical, chances are that
    you must change the paths.

    :return: 4 FreeLing components for pre-processing
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    pyfreeling.util_init_locale('default')
    tk = pyfreeling.tokenizer(config['FREELING']['Data'] + config['FREELING']['Lang'] + '/twitter/tokenizer.dat')
    sp = pyfreeling.splitter(config['FREELING']['Data'] + config['FREELING']['Lang'] + '/splitter.dat')
    umap = pyfreeling.RE_map(config['FREELING']['Data'] + config['FREELING']['Lang'] + '/twitter/usermap.dat')

    op = pyfreeling.maco_options("es")
    op.set_data_files("",
                      config['FREELING']['Data'] + "common/punct.dat",
                      config['FREELING']['Data'] + config['FREELING']['Lang'] + "/dicc.src",
                      config['FREELING']['Data'] + config['FREELING']['Lang'] + "/afixos.dat",
                      "",
                      config['FREELING']['Data'] + config['FREELING']['Lang'] + "/locucions.dat",
                      config['FREELING']['Data'] + config['FREELING']['Lang'] + "/np.dat",
                      config['FREELING']['Data'] + config['FREELING']['Lang'] + "/quantities.dat",
                      config['FREELING']['Data'] + config['FREELING']['Lang'] + "/probabilitats.dat")

    mf = pyfreeling.maco(op)
    mf.set_active_options(False, True, True, True,
                          True, True, False, True,
                          True, True, True, True)
    return tk, sp, umap, mf


def is_proper(t):
    """
    Filter for proper nouns in the tokens
    :param t: the POS tag of the token
    :return: True if FreeLing tagged the token as Noun and Proper, False otherwise
    """
    return t[0:2]=='NP'


def is_sw(w):
    """
    Filter for stopwords in the tokens

    :param w: the word of the token
    :return: True if the word is in the list of stopwords from NLTK, False otherwise
    """
    return w.lower() in stopwords.words('spanish')


def is_punct(w):
    """
    Filter for tokens that are basically punctuation or special symbols

    :param w: the word of the token
    :return: True if the word is punctuation, False otherwise
    """
    from string import punctuation
    for p in punctuation:
        w = w.replace(p, '')
    for n in '0123456789':
        w = w.replace(n, '')
    # If, after removing all special symbols and numbers, the remaining is 0-3 characters long, then it's filtered.
    return is_short(w)


def is_short(w):
    """
    Filter for short tokens

    :param w: the word of the token
    :return: True if the word is short, False otherwise
    """
    import emoji
    if w.startswith('@') or w.startswith('#') or w.startswith('https://'):
        # Mentions, Hashtags and URLs are also considered short words
        return True
    elif len(w) == 1:
        # Emojis are NOT considered short words
        return w[0] not in emoji.UNICODE_EMOJI
    else:
        # Words shorter than 3 words are considered short
        return len(w) < 3


def preprocess(tweets):
    """
    Pre-process the dataset: tokenizes, lemmatizes, and analyzes it morphologically to determine its POS tags.
    :param tweets: the dataset as a list of strings (the tweets)
    :return: the dataset already processed i.e. the tweets as tokens
    """
    lemmas = []
    tk, sp, umap, mf = setup_freeling()
    for tw in tweets:
        tokens = tk.tokenize(tw)
        tokens = sp.split(tokens)
        tokens = umap.analyze(tokens)
        tokens = mf.analyze(tokens)

        # Only keep those tokens that pass certain filters
        lemmas.append([x.get_lemma() for sent in tokens for x in sent
                       if not is_proper(x.get_tag())
                       and not is_sw(x.get_lemma())
                       and not is_punct(x.get_lemma())
                       and not is_short(x.get_lemma())])
    return lemmas
