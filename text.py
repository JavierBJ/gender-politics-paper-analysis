import pyfreeling
import configparser


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


def preprocess(tweets, label):
    maxlen=0
    ls_tokens = []
    labels = []
    tk, sp, umap, mf = setup_freeling()
    for tw, l in zip(tweets['full_text'], tweets[label]):
        tokens = tk.tokenize(tw)
        if len(tokens)>maxlen:
            maxlen = len(tokens)
        tokens = sp.split(tokens)
        tokens = umap.analyze(tokens)
        tokens = mf.analyze(tokens)
        ls_tokens.append([(x.get_form(), x.get_lemma(), x.get_tag()) for sent in tokens for x in sent])
        labels.append(l)
    return ls_tokens, labels
