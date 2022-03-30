"""
    plot decision tree
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, recall_score, roc_curve
from sklearn.tree import export_graphviz

import os

import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')


def grid_search(train_data, model, parameters):
    for parameter in parameters:
        cv_fold = StratifiedKFold(n_splits=10, random_state=20211001, shuffle=True)
        gsv = GridSearchCV(model, param_grid=parameter, scoring='accuracy', cv=cv_fold, return_train_score=True)
        gsv.fit(train_data[:, 1:], train_data[:, 0])
        model = gsv.best_estimator_
        print('The best parameters are', gsv.best_params_,
              'with train_acc: ', max(gsv.cv_results_['mean_train_score']),
              'and validation_acc:', max(gsv.cv_results_['mean_test_score']))

    return model


def compute_metrics(model, train_data, test_data):
    train_x, train_y = train_data[:, 1:], train_data[:, 0]
    test_x, test_y = test_data[:, 1:], test_data[:, 0]
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    prob_y = model.predict_proba(test_x)[:, 1]
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()

    tprs_, fprs_ = [], np.linspace(0, 1, 101)
    fpr_, tpr_, threshold = roc_curve(test_y, prob_y)
    tprs_.append(np.interp(fprs_, fpr_, tpr_))
    tprs_[0][0], tprs_[0][-1] = 0, 1

    metrics = dict()
    metrics['train_acc'] = accuracy_score(train_y, model.predict(train_x))
    metrics['test_acc'] = accuracy_score(test_y, pred_y)
    metrics['test_sens'] = recall_score(test_y, pred_y)
    metrics['test_spec'] = tn / (tn + fp)
    metrics['test_auc'] = roc_auc_score(test_y, prob_y)
    metrics['test_ppv'] = tp / (tp + fp)
    metrics['test_npv'] = tn / (tn + fn)

    return metrics, np.c_[test_y, prob_y], tprs_[0]


if __name__ == '__main__':
    seed = 21211105
    is_plot_roc = False

    data = pd.read_excel('../data.xlsx', sheet_name='mimic', index_col=0).\
        drop(['age', 'gender', 'ORG_NAME', 'LABEL_NAME'], axis=1)
    cols = ['label', 'WBC', 'AST', 'ALT', 'TBIL', 'AKP', 'BASO', 'EOS', 'NEUT', 'Lac', 'LDH', 'BEecf']
    data = data[cols]
    fea_names = data.columns.tolist()[1:]

    labels, features = data['label'].values.astype(np.uint8), data.drop(['label'], axis=1).values
    x_train, x_test, y_train, y_test = \
        train_test_split(features, labels, random_state=seed * 7, test_size=0.2, stratify=labels)
    scale = StandardScaler()
    scale.fit(x_train)
    x_train, x_test = scale.transform(x_train), scale.transform(x_test)
    train, test = np.c_[y_train, x_train], np.c_[y_test, x_test]
    print(data.shape, train.shape, test.shape)

    clf = DecisionTreeClassifier(random_state=seed*13, class_weight='balanced', max_features=len(cols)-1)
    metrics_base, _, _ = compute_metrics(clf, train, test)
    print(metrics_base)
    paras_tree = [{'max_depth': np.arange(3, 9, 1), 'min_samples_leaf': np.arange(0.08, 0.09, 0.001)}]
    clf = grid_search(train, clf, paras_tree)
    metrics_MIMIC, true_prob, _ = compute_metrics(clf, train, test)
    print('MIMIC：', metrics_MIMIC)

    test = pd.read_excel('../data.xlsx', sheet_name='ql', index_col=0)
    test = test[cols].values
    test = np.c_[test[:, 0], scale.transform(test[:, 1:])]
    metrics_QL, true_prob_ql, tprs = compute_metrics(clf, train, test)
    print('QL：', metrics_QL)

    clf.fit(data.values[:, 1:], data.values[:, 0].astype(np.uint8))

    dot_data = export_graphviz(clf, out_file='../figs/tree.dot',
                               feature_names=fea_names, class_names=['gram-neg', 'gram-pos'], label='all', filled=True,
                               leaves_parallel=False, impurity=False, node_ids=False, proportion=False,
                               rotate=False, rounded=True, precision=2)
    os.system('dot -Tpng ../figs/tree.dot'
              '-o ../figs/tree.dot/tree.png')
