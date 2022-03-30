"""
    model train and test in MIMIC database
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time
import matplotlib.pyplot as plt


def grid_search(train_data, model, parameters):
    for parameter in parameters:
        cv_fold = StratifiedKFold(n_splits=10, random_state=20211001, shuffle=True)
        gsv = GridSearchCV(model, param_grid=parameter, scoring='accuracy', cv=cv_fold, return_train_score=True)
        gsv.fit(train_data[:, 1:], train_data[:, 0])
        model = gsv.best_estimator_
        print('The best parameters are', gsv.best_params_,
              'with train_acc: ', max(gsv.cv_results_['mean_train_score']),
              'and validation_acc:', max(gsv.cv_results_['mean_test_score']))
        # plot_valid_curve(model, train_data, parameter)

    return model


def compute_metrics(model, train_data, test_data):
    train_x, train_y = train_data[:, 1:], train_data[:, 0]
    test_x, test_y = test_data[:, 1:], test_data[:, 0]
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    prob_y = model.decision_function(test_x) if 'decision_function' in model.__dir__() \
        else model.predict_proba(test_x)[:, 1]
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()

    metrics = dict()
    metrics['train_acc'] = accuracy_score(train_y, model.predict(train_x))
    metrics['test_acc'] = accuracy_score(test_y, pred_y)
    metrics['test_sens'] = recall_score(test_y, pred_y)
    metrics['test_spec'] = tn / (tn + fp)
    metrics['test_auc'] = roc_auc_score(test_y, prob_y)

    return metrics


def plot_learning_curve(model, train_data):
    train_size, train_scores, test_scores = learning_curve(model, train_data[:, 1:], train_data[:, 0],
                                                           train_sizes=np.linspace(0.1, 1, 10), scoring='accuracy')
    train_score_mean, train_score_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_score_mean, test_score_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

    plt.title('Learning Curve')
    plt.xlabel('training examples')
    plt.ylabel('score')
    # plt.ylim(0.0, 1.1)
    plt.grid()
    plt.fill_between(train_size, train_score_mean - train_score_std, train_score_mean + train_score_std,
                     color='r', alpha=0.2, lw=2)
    plt.fill_between(train_size, test_score_mean - test_score_std, test_score_mean + test_score_std,
                     color='g', alpha=0.2, lw=2)
    plt.plot(train_size, train_score_mean, 'o-', color='r', label='training score')
    plt.plot(train_size, test_score_mean, 'o-', color='g', label='CV score')
    plt.legend(loc='best')
    plt.show()


def plot_valid_curve(model, train_data, params):
    x, y = train_data[:, 1:], train_data[:, 0]
    for param in params.items():
        train_scores, test_scores = \
            validation_curve(model, x, y, param_name=param[0], param_range=param[1], scoring='accuracy')
        train_score_mean, train_score_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
        test_score_mean, test_score_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

        plt.title('validation_curve of %s' % param[0])
        plt.xlabel(param[0])
        plt.ylabel('score')
        # plt.ylim(0.5, 1.1)
        plt.grid()
        plt.fill_between(param[1], train_score_mean - train_score_std, train_score_mean + train_score_std,
                         color='r', alpha=0.2, lw=2)
        plt.fill_between(param[1], test_score_mean - test_score_std, test_score_mean + test_score_std,
                         color='g', alpha=0.2, lw=2)
        plt.plot(param[1], train_score_mean, 'o-', color='r', label='training score')
        plt.plot(param[1], test_score_mean, 'o-', color='g', label='cross-validation score')
        plt.legend(loc='best')
        plt.show()


def data_scale(train_data, test_data):
    scale = StandardScaler()
    scale.fit(train_data)
    train_data = scale.transform(train_data)
    test_data = scale.transform(test_data)

    return train_data, test_data


if __name__ == '__main__':
    start = time.time()
    print("start searching at ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)))

    seeds = [20191122, 4200526, 12240201, 20211211, 19961997]
    data = pd.read_excel('../data.xlsx', sheet_name='mimic', index_col=0)
    labels, features = data['label'].values, data.drop(['label'], axis=1).values
    print(data.shape)

    res = pd.DataFrame(index=seeds, columns=['train_acc', 'test_acc', 'test_sens', 'test_spec', 'test_auc', 'params'])
    for seed in seeds:
        print("==================== seed =", seed, '=================================')
        x_train, x_test, y_train, y_test = \
            train_test_split(features, labels, random_state=seed*7, test_size=0.2, stratify=labels)
        x_train, x_test = data_scale(x_train, x_test)
        train = np.c_[y_train, x_train]
        test = np.c_[y_test, x_test]

        rf = RandomForestClassifier(n_jobs=2, random_state=seed*13, class_weight='balanced')
        svm = SVC(class_weight='balanced')
        lr = LogisticRegression(n_jobs=2, solver='saga', random_state=seed*13, penalty='elasticnet', l1_ratio=0,
                                class_weight='balanced', max_iter=2000)
        mlp = MLPClassifier(random_state=seed*13, max_iter=2000, early_stopping=True)
        print('基模型分类效果：', compute_metrics(svm, train, test))

        paras_rf = [{'n_estimators': np.arange(50, 400, 50)},
                    {'max_depth': np.arange(5, 20, 2), 'min_samples_split': np.arange(0.01, 0.02, 0.002)},
                    {'min_samples_split': np.arange(0.01, 0.02, 0.002), 'min_samples_leaf': np.arange(0.01, 0.02, 0.002)},
                    {'max_features': np.arange(1, int(np.sqrt(train.shape[1] - 1)), 1)},
                    {'n_estimators': np.arange(100, 1100, 100)}]
        paras_svm = [{'C': np.arange(0.01, 4, 0.3), 'gamma': np.arange(1e-5, 1e-3, 1e-4)}]
        paras_lr = [{'C': np.arange(0.1, 1.1, 0.1), 'l1_ratio': np.arange(0, 1.1, 0.1)}]
        paras_mlp = [{'hidden_layer_sizes': [(4,), (8,), (16,), (32,), (64,),
                                             (4, 4), (8, 8), (16, 16), (32, 32), (64, 64),
                                             (8, 4), (16, 8), (16, 4), (32, 16), (32, 8), (64, 32), (64, 16)]},
                     {'alpha': np.arange(1e-5, 1e-3, 1e-4)}]
        clf = grid_search(train, svm, paras_rf)
        metric_adjust = compute_metrics(clf, train, test)

        keys = set()
        for paras in paras_svm:
            for key in paras.keys():
                keys.add(key)
        paras = dict.fromkeys(keys)
        for key in keys:
            paras[key] = clf.get_params()[key]
        print('调参后模型参数：', paras)
        print('调参后的分类效果：', metric_adjust)
        # plot_learning_curve(clf, train)

        res.loc[seed]['params'] = paras
        res.loc[seed]['train_acc'] = metric_adjust['train_acc']
        res.loc[seed]['test_acc'] = metric_adjust['test_acc']
        res.loc[seed]['test_sens'] = metric_adjust['test_sens']
        res.loc[seed]['test_spec'] = metric_adjust['test_spec']
        res.loc[seed]['test_auc'] = metric_adjust['test_auc']

    res.to_excel('../res/res-test.xlsx')

    end = time.time()
    print("finish searching at ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end)))
    print("time-expense:", (end - start) / 60, "minutes")
