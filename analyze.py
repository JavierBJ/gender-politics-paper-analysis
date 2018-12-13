import text
import os, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def results_to_file(results, names, to, top=50):
    to = open(to, 'w')
    l = [(names[x], v) for (x, v) in enumerate(results)]
    males = sorted(l, key=lambda x: x[1], reverse=True)[:top]
    females = sorted(l, key=lambda x: x[1])[:top]
    print('Male predictors:', file=to)
    print('word : coef : odds_ratio', file=to)
    for (word, value) in males:
        print(word, ':', value, ':', np.exp(value), file=to)
    print('\nFemale predictors:', file=to)
    print('word : -coef : odds_ratio', file=to)
    for (word, value) in females:
        print(word, ':', -value, ':', np.exp(-value), file=to)


def train_and_evaluate(X, y, X_test, y_test, model, folds):
    kf = KFold(n_splits=folds)
    results = np.array((X.get_shape()[1],))
    train_score = 0
    val_score = 0
    for train_idx, val_idx in kf.split(X):
        model = model.fit(X[train_idx], y[train_idx])
        tr_preds = model.predict(X[train_idx])
        val_preds = model.predict(X[val_idx])
        train_score += metrics.roc_auc_score(y[train_idx], tr_preds)
        val_score += metrics.roc_auc_score(y[val_idx], val_preds)
        results = np.add(results, model.coef_.flatten(), casting='unsafe')
    results /= folds
    train_score /= folds
    val_score /= folds

    # Re-train with the whole training set
    model = model.fit(X, y)
    results = model.coef_.flatten()

    # Let test scores calculated, though they must not be observed during fine-tuning
    te_preds = model.predict(X_test)
    test_score = metrics.roc_auc_score(y_test, te_preds)

    conf = metrics.confusion_matrix(y_test, te_preds)
    scores = {'Training':train_score, 'Validation':val_score, 'Test':test_score}
    return results, scores, conf


def analyze(path, alg='lasso', alpha=1, kfolds=10, kwr=5000, sw=True):
    # Craft path for outputs from parameters
    out_path = 'out/'
    expname = path.split('/')[-1].split('.')[0] + '_' + alg
    if kwr is not None:
        expname += '_rank' + str(kwr)
    if sw:
        expname += '_sw'
    out_path += expname
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    log = open(out_path + '/log.txt', 'w')

    # Load CSV from path
    tweets_df = pd.read_csv(path, delimiter=';', encoding='utf-8')
    label = tweets_df['gender']
    tweets_df = tweets_df['tweet']

    # Pre-process and feature extraction
    tweets_df, label = text.preprocess(tweets_df, label)
    y = np.array(label).reshape((-1,))
    x_train, x_test, y_train, y_test = train_test_split(tweets_df, y, test_size=0.2)

    # Format required by default CountVectorizer is sequence of strings separated by e.g. spaces
    x_train = [' '.join(x) for x in x_train]
    x_test = [' '.join(x) for x in x_test]
    feat_ext = CountVectorizer(encoding='utf-8', token_pattern=' ', ngram_range=(1,1), analyzer='word',
                               max_features=kwr, binary=True)
    x_train = feat_ext.fit_transform(x_train)
    x_test = feat_ext.transform(x_test)

    best_val = 0
    best_test = 0
    for a in alpha:
        c_reg = 1 / a
        if alg in ['lasso', 'l1']:
            model = LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1', C=c_reg)
        elif alg in ['ridge', 'l2']:
            model = LogisticRegression(class_weight='balanced', C=c_reg)
        elif alg == 'svm':
            model = LinearSVC(class_weight='balanced', C=c_reg)

        # Run experiments
        results, scores, confusion = train_and_evaluate(x_train, y_train, x_test, y_test, model, kfolds)

        # Write to results file
        os.makedirs(out_path, exist_ok=True)
        out_path += '/' + 'a' + str(a).replace('.', '_') + '.txt'
        results_to_file(results=results, names=feat_ext.get_feature_names(), to=out_path, top=500)

        # Write to log file
        print('Experiment', expname, ', case', a, '...', file=log)
        print('Training score:', scores['Training'])
        print('Validation score:', scores['Validation'])
        print('', file=log)

        if scores['Validation'] > best_val:
            best_val = scores['Validation']
            best_test = scores['Test']
            best_a = a

    print('Best experiment: Alpha =', best_a, file=log)
    print('Test score:', best_test, file=log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run experiments of language analysis.')
    parser.add_argument('path',
                        help='path to csv file with dataset')
    parser.add_argument('alg', choices=['lasso', 'l1', 'ridge', 'l2', 'svm'], help='learning algorithm')
    parser.add_argument('-a', '--alpha', type=float, nargs='+', default=1.0, help='regularization parameters to try')
    parser.add_argument('-k', '--kfolds', type=int, default=10, help='k for k-fold cross-validation')
    parser.add_argument('-r', '--keepwordsrank', type=int,
                        help='number of features to be used ranked by number of appearances')
    parser.add_argument('-s', '--stopwords', action='store_true', help='remove stopwords if active')
    args = parser.parse_args()

    analyze(args.path, args.alg, args.alpha, args.kfolds, args.keepwordsrank, args.stopwords)
