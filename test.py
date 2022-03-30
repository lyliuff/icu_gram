"""
    model test in Qilu Hospital cohort
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
pd.set_option('display.max_columns', None)


def data_scale(train_data, test_data):
    scale = StandardScaler()
    scale.fit(train_data[:, 1:])
    train_data_x = scale.transform(train_data[:, 1:])
    test_data_x = scale.transform(test_data[:, 1:])

    return np.c_[train_data[:, 0], train_data_x], np.c_[test_data[:, 0], test_data_x]


def model_verify(clf, dataset):
    x, y = dataset.drop(['label'], axis=1), dataset['label']
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, random_state=seed * 7, test_size=0.2, stratify=y)
    scale = StandardScaler()
    scale.fit(x_train)
    x_train, x_test = scale.transform(x_train), scale.transform(x_test)

    clf.fit(x_train, y_train)
    pred_y = clf.predict(x_test)
    prob_y = clf.decision_function(x_test) if 'decision_function' in clf.__dir__() else clf.predict_proba(x_test)[:, 1]

    metrics = dict()
    metrics['acc'] = accuracy_score(y_test, pred_y)
    metrics['auc'] = roc_auc_score(y_test, prob_y)

    return metrics


def model_test(clf, train_data, test_data):
    train_x, train_y = train_data[:, 1:], train_data[:, 0]
    test_x, test_y = test_data[:, 1:], test_data[:, 0]

    clf.fit(train_x, train_y)

    metrics = dict()
    train_pred_y = clf.predict(train_x)
    train_prob_y = clf.decision_function(train_x) if 'decision_function' in clf.__dir__() \
        else clf.predict_proba(train_x)[:, 1]
    tn, fp, fn, tp = confusion_matrix(train_y, train_pred_y).ravel()
    metrics['train_auc'] = roc_auc_score(train_y, train_prob_y)
    metrics['train_acc'] = accuracy_score(train_y, train_pred_y)
    metrics['train_sens'] = recall_score(train_y, train_pred_y)
    metrics['train_spec'] = tn / (tn + fp)
    metrics['train_ppv'] = tp / (tp + fp)
    metrics['train_npv'] = tn / (tn + fn)

    test_prob_y = clf.decision_function(test_x) if 'decision_function' in clf.__dir__() \
        else clf.predict_proba(test_x)[:, 1]
    metrics['test_auc'] = roc_auc_score(test_y, test_prob_y)
    test_pred_y = clf.predict(test_x)
    tn, fp, fn, tp = confusion_matrix(test_y, test_pred_y).ravel()
    metrics['test_acc'] = accuracy_score(test_y, test_pred_y)
    metrics['test_sens'] = recall_score(test_y, test_pred_y)
    metrics['test_spec'] = tn / (tn + fp)
    metrics['test_ppv'] = tp / (tp + fp)
    metrics['test_npv'] = tn / (tn + fn)

    return metrics, np.int8(test_y != test_pred_y), test_prob_y


def plot_roc(tpr_clf):
    fpr = np.linspace(0, 1, 101)
    plt.figure(dpi=500, figsize=(4.2, 3.8))
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='-.')
    plt.plot(fpr, tpr_clf, lw=2, color='red', label='AUC = %s' % auc(fpr, tpr_clf))
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('ROC Curves of RF', fontsize=18)
    plt.legend(loc="lower right", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if is_save:
        plt.savefig('../figs/ROC_ql.png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_roc_curve(clf, train_data, test_data):
    train_x, train_y = train_data[:, 1:], train_data[:, 0]
    test_x, test_y = test_data[:, 1:], test_data[:, 0]

    clf.fit(train_x, train_y)
    prob_y = clf.decision_function(test_x) if 'decision_function' in clf.__dir__() \
        else clf.predict_proba(test_x)[:, 1]

    tprs_, fprs_ = [], np.linspace(0, 1, 101)
    fpr_, tpr_, threshold = roc_curve(test_y, prob_y)
    tprs_.append(np.interp(fprs_, fpr_, tpr_))
    tprs_[0][0], tprs_[0][-1] = 0, 1
    plot_roc(tprs_[0])


if __name__ == '__main__':
    is_save = False
    seed = 19961997
    params_rf = {'min_samples_leaf': 0.02, 'max_features': 2, 'n_estimators': 250,
                 'min_samples_split': 0.02, 'max_depth': 8}

    train = pd.read_excel('../data.xlsx', sheet_name='mimic', index_col=0)
    test = pd.read_excel('../data.xlsx', sheet_name='ql', index_col=0)
    print('seed =', seed, ', train shape:', train.shape, ', test shape:', test.shape)

    rf = RandomForestClassifier(n_jobs=2, random_state=seed * 13, class_weight='balanced', **params_rf)
    print('model verification:', model_verify(rf, train))

    train, test = data_scale(train.values, test.values)
    metrics_test, pred_test, prob_test_rf = model_test(rf, train, test)
    print(metrics_test)

    plot_roc_curve(rf, train, test)
