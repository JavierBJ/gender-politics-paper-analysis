import text
import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

score_names = {'tr_f1':'Train_F1', 'val_f1':'Validation_F1', 'te_f1':'Test_F1', \
               'tr_prec':'Train_Precision', 'val_prec':'Validation_Precision', 'te_prec':'Test_Precision', \
               'tr_rec':'Train_Recall', 'val_rec':'Validation_Recall', 'te_rec':'Test_Recall', \
               'tr_auc':'Train_AUC', 'val_auc':'Validation_AUC', 'te_auc':'Test_AUC'}


def results_to_file(top=50, to=sys.stdout):
    l = [(self.extractor.features_idx[x], v, self.extractor.supports[self.extractor.features_idx[x]]) for (x, v) in
         enumerate(self.results)]
    males = sorted(l, key=lambda x: x[1], reverse=True)[:top]
    females = sorted(l, key=lambda x: x[1])[:top]
    print('Male predictors:', file=to)
    print('word : coef : odds_ratio', file=to)
    for (word, value, sup) in males:
        print(word, ':', value, ':', np.exp(value), ':', sup, file=to)
    print('\nFemale predictors:', file=to)
    print('word : -coef : odds_ratio', file=to)
    for (word, value, sup) in females:
        print(word, ':', -value, ':', np.exp(-value), ':', sup, file=to)


def train_and_evaluate(X, y, X_test, y_test, model, folds):
    kf = KFold(n_splits=folds)
    results = np.array((X.get_shape()[1],))
    tr_f1, tr_prec, tr_recall, tr_auc, val_f1, val_prec, val_recall, val_auc = 0, 0, 0, 0, 0, 0, 0, 0
    for train_idx, val_idx in kf.split(X):
        model = model.fit(X[train_idx], y[train_idx])
        tr_preds = model.predict(X[train_idx])
        val_preds = model.predict(X[val_idx])
        tr_f1 += metrics.f1_score(y[train_idx], tr_preds)
        tr_prec += metrics.precision_score(y[train_idx], tr_preds)
        tr_recall += metrics.recall_score(y[train_idx], tr_preds)
        tr_auc += metrics.roc_auc_score(y[train_idx], tr_preds)
        val_f1 += metrics.f1_score(y[val_idx], val_preds)
        val_prec += metrics.precision_score(y[val_idx], val_preds)
        val_recall += metrics.recall_score(y[val_idx], val_preds)
        val_auc += metrics.roc_auc_score(y[val_idx], val_preds)
        results = np.add(results, model.coef_.flatten(), casting='unsafe')
    results /= folds
    tr_f1 /= folds
    tr_prec /= folds
    tr_recall /= folds
    tr_auc /= folds
    val_f1 /= folds
    val_prec /= folds
    val_recall /= folds
    val_auc /= folds

    # Re-train with the whole training set
    model = model.fit(X, y)
    results = model.coef_.flatten()

    # Let test scores calculated, though they must not be observed during fine-tuning
    te_preds = model.predict(X_test)
    te_preds = te_preds
    te_f1 = metrics.f1_score(y_test, te_preds)
    te_prec = metrics.precision_score(y_test, te_preds)
    te_recall = metrics.recall_score(y_test, te_preds)
    te_auc = metrics.roc_auc_score(y_test, te_preds)

    conf = metrics.confusion_matrix(y_test, te_preds)
    scores = {'Train_F1': tr_f1, 'Train_Precision': tr_prec, 'Train_Recall': tr_recall, 'Train_AUC': tr_auc, \
                   'Validation_F1': val_f1, 'Validation_Precision': val_prec, 'Validation_Recall': val_recall,
                   'Validation_AUC': val_auc, \
                   'Test_F1': te_f1, 'Test_Precision': te_prec, 'Test_Recall': te_recall, 'Test_AUC': te_auc}
    return scores


def analyze(path, alg='lasso', score='val_auc', limit=0, kfolds=10, sw=True, kwr=5000, alpha=1):
    # Craft path for outputs from parameters
    out_path = 'results/analysis/'
    expname = path.split('.')[0] + '_' + alg
    if kwr is not None:
        expname += '_rank' + str(kwr)
    if sw:
        expname += '_sw'
    out_path += expname
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    log = open(out_path + '/log.txt', 'w')

    # Translates short names of scores into long names understood by the model
    if score not in score_names.values():
        score = score_names[score]

    # Load CSV from path
    tweets_df = pd.read_csv(path, delimiter=';', encoding='utf-8')
    label = tweets_df['gender']
    tweets_df = tweets_df.drop('gender')

    # Preprocess and feature extraction
    tweets_df, label = text.preprocess(tweets_df, label)
    y = np.array(label).reshape((-1,))
    X_train, X_test, y_train, y_test = train_test_split(tweets_df, y, test_size=0.2)

    feat_ext = CountVectorizer(encoding='utf-8', token_pattern=' ', ngram_range=(1,1), analyzer='word',
                               max_features=kwr, binary=True)
    X_train = feat_ext.fit_trasform(X_train)
    X_test = feat_ext.transform(X_test)

    best = 0
    for a in alpha:
        c_reg = 1 / a
        if alg in ['lasso', 'l1']:
            model = LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1', C=c_reg)
        elif alg in ['ridge', 'l2']:
            model = LogisticRegression(class_weight='balanced', C=c_reg)
        elif alg == 'svm':
            model = LinearSVC(class_weight='balanced', C=c_reg)

        # Run experiments
        # variables: res, analyzer
        res = train_and_evaluate(X_train, y_train, X_test, y_test, model, kfolds)

        # Write to results file
        os.makedirs(out_path, exist_ok=True)
        results_to_file(top=500, to=open(out_path + '/' + 'a' + str(a).replace('.', '_') + '.txt', 'w'))

        # Write to log file
        print('Experiment', expname, ', case', a, '...', file=log)
        print('Train - Validation', file=log)
        print('F1:', res['Train_F1'], '-', res['Validation_F1'], file=log)
        print('Precision:', res['Train_Precision'], '-', res['Validation_Precision'], file=log)
        print('Recall:', res['Train_Recall'], '-', res['Validation_Recall'], file=log)
        print('AUC:', res['Train_AUC'], '-', res['Validation_AUC'], file=log)
        print('', file=log)

        if res[score] > best:
            best = res[score]
            best_analyzer = analyzer
            best_a = a

    print('Best experiment: Alpha =', best_a, file=log)
    print('Test scores', file=log)
    print('F1:', best_analyzer.scores['Test_F1'], file=log)
    print('Precision:', best_analyzer.scores['Test_Precision'], file=log)
    print('Recall:', best_analyzer.scores['Test_Recall'], file=log)
    print('AUC:', best_analyzer.scores['Test_AUC'], file=log)
