import argparse
import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run experiments of language analysis.')
    parser.add_argument('path',
                        help='path to csv file with dataset')
    parser.add_argument('alg', choices=['lasso', 'l1', 'ridge', 'l2', 'svm'], help='learning algorithm')
    parser.add_argument('score',
                        choices=['tr_f1', 'tr_prec', 'tr_rec', 'tr_auc', 'val_f1', 'val_prec', 'val_rec', 'val_auc'],
                        help='scoring function to validate')
    parser.add_argument('-t', '--limit', type=int, default=0, help='maximum number of tweets recovered (0 for all)')
    parser.add_argument('-k', '--kfolds', type=int, default=10, help='k for k-fold cross-validation')
    parser.add_argument('-s', '--stopwords', action='store_true', help='remove stopwords if active')
    parser.add_argument('-r', '--keepwordsrank', type=int,
                        help='number of features to be used ranked by number of appearances')
    parser.add_argument('-a', '--alpha', type=float, nargs='+', default=1.0, help='regularization parameters to try')

    args = parser.parse_args()
    print('path:', args.path)
    print('alg:', args.alg)
    print('score:', args.score)
    print('limit:', args.limit)
    print('kfolds:', args.kfolds)
    print('stopwords:', args.stopwords)
    print('keepwordsrank:', args.keepwordsrank)
    print('alpha:', args.alpha)

    model.analyze(args.path, args.alg, args.score, args.limit, args.kfolds, args.stopwords, args.keepwordsrank,
                  args.alpha)
