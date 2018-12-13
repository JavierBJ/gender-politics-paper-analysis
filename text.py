import pyfreeling
import configparser
from nltk.corpus import stopwords


def setup_freeling():
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
    return t[0:2]=='NP'


def is_sw(w):
    return w.lower() in stopwords.words('spanish')


def is_punct(w):
    from string import punctuation
    for p in punctuation:
        w = w.replace(p, '')
    for n in '0123456789':
        w = w.replace(n, '')
    return is_short(w)


def is_short(w):
    import emoji
    if w.startswith('@') or w.startswith('#') or w.startswith('https://'):
        return True
    elif len(w) == 1:
        return w[0] not in emoji.UNICODE_EMOJI
    else:
        return len(w) < 3


def preprocess(tweets, label):
    lemmas = []
    labels = []
    tk, sp, umap, mf = setup_freeling()
    for tw, l in zip(tweets['full_text'], tweets[label]):
        tokens = tk.tokenize(tw)
        tokens = sp.split(tokens)
        tokens = umap.analyze(tokens)
        tokens = mf.analyze(tokens)

        lemmas.append([x.get_lemma() for sent in tokens for x in sent
                       if not is_proper(x.get_tag())
                       and not is_sw(x.get_lemma())
                       and not is_punct(x.get_lemma())
                       and not is_short(x.get_lemma())])
        labels.append(l)
    return lemmas, labels
