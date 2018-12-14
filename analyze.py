"""gender-politics-paper-analysis: analyze.py

This module contains the main code for the language analysis by gender, as well as the functions that run the machine
learning experiments. It can be called from the command line using the following arguments:

python3 analyze.py path alg
    where <path> is the path to a CSV file containing two columns called "tweet" and "gender" in the header, and
    separated by <;>,
    and <alg> can be one of the following: lasso, l1, ridge, l2, svm. l1 is a shortcut for lasso, and l2 is a shortcut
    for ridge.
    Optional arguments are also available. For a complete description of the program please type python3 analyze.py -h

The code makes heavy use of well-known libraries from the ecosystem of scientific computing and data science in Python,
like numpy, pandas and scikit-learn. It also calls the module text.py, which contains the pre-processing steps using
the libraries FreeLing 4.0 and NLTK.

Author: Javier Beltran Jorba
Institut Barcelona d'Estudis Internacionals
Universitat Pompeu Fabra
"""


import text
import os, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def results_to_file(coefficients, names, to, top):
    """Prints the list of most predictive words for males and females in a file.

    :param coefficients: 1D array of coefficients of the best model found.
    :param names: list with the words corresponding to each coefficient, index-wise.
    :param to: path of the output file.
    :param top: number of most predictive words to print. The same number is printed for males and females.
    :return: nothing. Prints results to the filesystem.
    """
    to = open(to, 'w')
    l = [(names[x], v) for (x, v) in enumerate(coefficients)]
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
    """Trains a model with the given dataset, using k-fold cross validation and saving the scores (balanced accuracy) for
    training, validation and test. Test score is calculated here for commodity even though it is still unknown by
    the analyzer if this model configuration will be the best. It is not used for anything but being ready for printing
    once all models are checked.

    :param X: training dataset of tweets already preprocessed and converted into feature counts, i.e. a binary matrix
    of (n_tweets, n_features)
    :param y: 1D array of labels for the tweets in X (1 for males, -1 for females but any two values should work).
    :param X_test: test dataset.
    :param y_test: test labels.
    :param model: a sklearn supervised algorithm with fit(X) and predict(X) methods.
    :param folds: number of folds in the k-fold cross validation.
    :return: three data, 1) 1D array with the coefficients of the model fitted; 2) a dictionary with the training,
    validation and test scores accessible by keys 'Training', 'Validation' and 'Test' respectively; 3) confusion matrix.
    """

    # Divides training dataset into training and validation, using a different split for validation at each iteration.
    kf = KFold(n_splits=folds)
    train_score = 0
    val_score = 0
    for train_idx, val_idx in kf.split(X):
        # Fit model with training part
        model = model.fit(X[train_idx], y[train_idx])

        # Predict with two datasets and get balanced accuracy score
        tr_preds = model.predict(X[train_idx])
        val_preds = model.predict(X[val_idx])
        train_score += metrics.roc_auc_score(y[train_idx], tr_preds)
        val_score += metrics.roc_auc_score(y[val_idx], val_preds)

    # Average scores for the k folds.
    train_score /= folds
    val_score /= folds

    # Re-train with the whole training set
    model = model.fit(X, y)
    coefficients = model.coef_.flatten()

    # Let test scores calculated, though they must not be observed during fine-tuning
    te_preds = model.predict(X_test)
    test_score = metrics.roc_auc_score(y_test, te_preds)

    # Save evaluation measures: scores and confusion matrix
    conf = metrics.confusion_matrix(y_test, te_preds)
    scores = {'Training':train_score, 'Validation':val_score, 'Test':test_score}
    return coefficients, scores, conf


def analyze(path, alg, alpha, kfolds, rank, top):
    """Performs the machine learning experiments with the dataset, algorithm, and parameters selected; and writes the
    results in different files inside an out/ folder: a log file with the scores obtained by each configuration of the
    algorithm, and a file with the most predictive words for each configuration.

    :param path: path to the dataset in csv format, using ; as separator and with two columns, tweet and gender. A
    header should be present in the first line with these column names.
    :param alg: string with the name of the algorithm to be used (lasso (or l1), ridge (or l2), svm).
    :param alpha: regularization parameter of the learning algorithm (higher means more regularization).
    :param kfolds: k parameter of k-fold cross validation (in how many chunks to divide the dataset and how many
    repetitions to perform).
    :param rank: only the most frequent words will be used as features. This parameter chooses how many.
    :param top: only the most predictive words will be printed to a file. This parameter chooses how many.
    :return: nothing. Prints results to the filesystem.
    """

    # Create paths for writing results
    dir_path = 'out/'
    expname = path.split('/')[-1].split('.')[0] + '_' + alg
    dir_path += expname
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    log = open(dir_path + '/log.txt', 'w')

    # Load CSV from path
    tweets_df = pd.read_csv(path, delimiter=';', encoding='utf-8')
    label = tweets_df['gender']
    tweets_df = tweets_df['tweet']

    # Pre-process tweets and divide into training and test
    tweets_df = text.preprocess(tweets_df)
    y = np.array(label).reshape((-1,))
    x_train, x_test, y_train, y_test = train_test_split(tweets_df, y, test_size=0.2)

    # Format required by default CountVectorizer is sequence of strings separated by e.g. spaces
    x_train = [' '.join(x) for x in x_train]
    x_test = [' '.join(x) for x in x_test]
    feat_ext = CountVectorizer(encoding='utf-8', ngram_range=(1,1), analyzer='word', max_features=rank, binary=True)
    x_train = feat_ext.fit_transform(x_train)
    x_test = feat_ext.transform(x_test)

    best_val = 0
    best_test = 0
    for a in alpha:
        # Choose learning algorithm based on arguments
        c_reg = 1 / a
        if alg in ['lasso', 'l1']:
            model = LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1', C=c_reg)
        elif alg in ['ridge', 'l2']:
            model = LogisticRegression(class_weight='balanced', C=c_reg)
        elif alg == 'svm':
            model = LinearSVC(class_weight='balanced', C=c_reg)

        # Fit model and test it
        results, scores, confusion = train_and_evaluate(x_train, y_train, x_test, y_test, model, kfolds)

        # Write to results file
        out_path = dir_path + '/' + 'a' + str(a).replace('.', '_') + '.txt'
        results_to_file(coefficients=results, names=feat_ext.get_feature_names(), to=out_path, top=top)

        # Write to log file
        print('Experiment', expname, ', case', a, '...', file=log)
        print('Training score:', scores['Training'], file=log)
        print('Validation score:', scores['Validation'], file=log)
        print('', file=log)

        # Save best configuration of algorithm to present test score later
        if scores['Validation'] > best_val:
            best_val = scores['Validation']
            best_test = scores['Test']
            best_a = a

    # Write test score to log file
    print('Best experiment: Alpha =', best_a, file=log)
    print('Test score:', best_test, file=log)


if __name__ == '__main__':
    """Main code of the analyzer of language by gender. Reads execution arguments or informs the user if the arguments 
    are wrong.
    """
    parser = argparse.ArgumentParser(description='run experiments of language analysis.')
    parser.add_argument('path',
                        help='path to csv file with dataset')
    parser.add_argument('alg', choices=['lasso', 'l1', 'ridge', 'l2', 'svm'], help='learning algorithm')
    parser.add_argument('-a', '--alpha', type=float, nargs='+', default=[1.0], help='regularization parameters to try')
    parser.add_argument('-k', '--kfolds', type=int, default=10, help='k for k-fold cross-validation')
    parser.add_argument('-r', '--rankfeatures', type=int, default=5000,
                        help='number of features to be used ranked by number of appearances')
    parser.add_argument('-t', '--topfeatures', type=int, default=50,
                        help='number of word features to be printed in the output files')
    args = parser.parse_args()

    analyze(args.path, args.alg, args.alpha, args.kfolds, args.rankfeatures, args.topfeatures)
