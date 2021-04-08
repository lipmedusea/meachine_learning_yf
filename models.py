from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
# from sklearn.externals import joblib
from xgboost import XGBClassifier
from model_evalu import evalution_model, plot_importance
from catboost import CatBoostClassifier, CatBoostRegressor,Pool
from collections import Counter
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.externals import joblib
from gcforest.gcforest import GCForest
from GCFores import gcForest
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from scipy import sparse
import xgboost
import lightgbm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.naive_bayes import MultinomialNB, GaussianNB


def adaboost_model(x_train, x_test, y_train, y_test, df_xbtest, df_ybtest):

    print('-------------------adaboost-------------------------')
    cout = Counter(y_train)
    tt = cout[0] / cout[1] - 21
    sample_weigh = np.where(y_train == 0, 1, tt)

    clfs = AdaBoostClassifier(n_estimators=20,
                              learning_rate=1.0,
                              algorithm='SAMME.R',
                              random_state=0
                              )
    param_grid = {
        'n_estimators': range(23, 25),
        'learning_rate': [0.1],
        'algorithm': ["SAMME.R"]
                 }
    dt_score = make_scorer(f1_score, pos_label=1)
    clfs = GridSearchCV(estimator=clfs,
                        param_grid=param_grid,
                        scoring=dt_score,
                        cv=5,
                        n_jobs=-1)

    clfs = clfs.fit(x_train, y_train.astype(int), sample_weight=sample_weigh)

    print("================训练集================")
    evalution_model(clfs, x_train, y_train)
    print("================测试集================")
    evalution_model(clfs, x_test, y_test)
    print("===========b_test===================")
    evalution_model(clfs, df_xbtest, df_ybtest)
    print("================Importance================")
    plot_importance(clfs, x_train.columns,
                    title='feature_importancet', n=30, method=0)


def lr_model(x_train,x_test,y_train,y_test,df_xbtest,df_ybtest):
    print('-------------------LR-------------------------')
    cout = Counter(y_train)
    tt = cout[0] / cout[1] - 26
    sample_weigh = np.where(y_train == 0, 1, tt)
    
    clfs = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, max_iter=100, multi_class='ovr',
                       penalty='l2', random_state=None, tol=0.0001, solver='liblinear',
                       verbose=0, warm_start=False)

    param_grid = {
        'C': [1, 0.1, 0.01],
        'penalty': ['l2'],
        'max_iter': [30, 50, 100,]
    }

    dt_score = make_scorer(f1_score, pos_label=1)
    clfs = GridSearchCV(estimator=clfs,
                        param_grid=param_grid,
                        scoring=dt_score,
                        cv=5,
                        n_jobs=-1)

    clfs = clfs.fit(x_train, y_train.astype(int), sample_weight = sample_weigh)


    print("================训练集================")
    evalution_model(clfs, x_train, y_train)
    print("================测试集================")
    evalution_model(clfs, x_test, y_test)
    print("===========b_test===================")
    evalution_model(clfs, df_xbtest,  df_ybtest)


def rf_mdoel(x_train, x_test, y_train, y_test, df_xbtest, df_ybtest):
    print('-------------------Rf-------------------------')

    cout = Counter(y_train)
    tt = cout[0] / cout[1] - 23
    sample_weigh = np.where(y_train == 0, 1, tt)
    clfs = RandomForestClassifier(n_estimators=24, max_depth=5, max_features=5, random_state=5, n_jobs=-1,
                                  # class_weight={0: 1, 1: tt}
                                  )
    param_grid = {
        'criterion': ['gini'],
        'n_estimators': range(20, 25),
        'max_depth': range(5, 7),
        'max_features': range(8, 10),
        # 'class_weight': [{1: i} for i in np.linspace(tt, tt+1, 1)]
                 }
    dt_score = make_scorer(f1_score, pos_label=1)
    make_scorer(f1_score, pos_label=1)
    clfs = GridSearchCV(estimator=clfs,
                        param_grid=param_grid,
                        scoring=dt_score,
                        cv=5,
                        n_jobs=-1)

    clfs = clfs.fit(x_train, y_train.astype(int), sample_weight=sample_weigh)
    b_pred = clfs.predict(df_xbtest)
    pre = round(precision_score(df_ybtest, b_pred, pos_label=1), 2)*100
    rec = round(recall_score(df_ybtest, b_pred, pos_label=1), 2)*100
    model_name = "models/rf_{}_{}.m".format(int(pre), int(rec))
    #joblib.dump(clfs, model_name)
    print(clfs.best_params_)


    print("================训练集================")
    evalution_model(clfs, x_train, y_train)
    print("================测试集================")
    evalution_model(clfs, x_test, y_test)
    print("===========b_test===================")
    evalution_model(clfs, df_xbtest,
                    df_ybtest)
    print("================Importance================")
    plot_importance(clfs, x_train.columns,
                    title='feature_importancet', n=30, method=0)


def gbdt_mdoel(x_train,x_test,y_train,y_test,df_xbtest,df_ybtest):
    print('-------------------GBDT-------------------------')
    clfs = GradientBoostingClassifier(loss="deviance", learning_rate=0.01,
                                      n_estimators=20, subsample=1.0,
                                      criterion="friedman_mse",
                                      min_samples_split=2, min_samples_leaf=1,
                                      max_depth=5, random_state=5)
    cout = Counter(y_train)
    tt = cout[0] / cout[1] - 23
    sample_weigh = np.where(y_train==0, 1, tt)
    param_grid = {"loss": ["deviance"],                   # GBDT parameters
                  # "learning_rate": [0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0],
                  "learning_rate": [0.1],
                  "n_estimators": range(20, 22),
                  "subsample": [0.8, 1.0],
                  "criterion": ["friedman_mse"],
                  "max_features": range(7, 10),       # DT parameters
                  "max_depth": range(5, 8),
                  # "min_samples_split": range(2, 10),
                  # "min_samples_leaf": range(1, 10),
                  # "min_weight_fraction_leaf": [0.08],    # Un-Tested parameters
                  "warm_start": [True],
                  # "min_impurity_decrease": [None],
                  # "max_leaf_nodes": [None],
                  }
    dt_score = make_scorer(f1_score, pos_label=1)
    make_scorer(f1_score, pos_label=1)
    clfs = GridSearchCV(estimator=clfs,
                        param_grid=param_grid,
                        scoring=dt_score,
                        cv=5,
                        n_jobs=-1)

    clfs = clfs.fit(x_train, y_train.astype(int),
                    sample_weight=sample_weigh
                    )
    print(clfs.best_params_)

    b_pred = clfs.predict(df_xbtest)
    pre = round(precision_score(df_ybtest, b_pred, pos_label=1), 2) * 100
    rec = round(recall_score(df_ybtest, b_pred, pos_label=1), 2) * 100
    model_name = "models/gbdt_{}_{}.m".format(int(pre), int(rec))
    #joblib.dump(clfs, model_name)

    print("================训练集================")
    evalution_model(clfs, x_train, y_train)
    print("================测试集================")
    evalution_model(clfs, x_test, y_test)
    print("===========b_test===================")
    evalution_model(clfs, df_xbtest,
                    df_ybtest)
    print("================Importance================")
    plot_importance(clfs, x_train.columns,
                    title='feature_importancet', n=30, method=0)


def lgb_sk_mdoel(x_train, x_test, y_train, y_test, df_xbtest, df_ybtest):
    print('-------------------LGB_SK-------------------------')

    clfs = lgb.LGBMClassifier(random_state=5,class_weight={1:4.16})
    cout = Counter(y_train)
    tt = cout[0]/cout[1]
    sample_weigh = np.where(y_train == 0, 1, tt)
    param_grid = {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": range(20, 25),
                    # "max_depth ": range(5, 6),
                     # "boosting_type": ["gbdt", "rf"],
                     "class_weight": [{1: i} for i in np.linspace(tt, tt+1, 1)],
                     "subsample": [1, 0.85]
                  }
    dt_score = make_scorer(f1_score, pos_label=1)
    clfs = GridSearchCV(estimator=clfs,
                        param_grid=param_grid,
                        scoring=dt_score,
                        cv=5,
                        n_jobs=-1)

    clfs = clfs.fit(x_train, y_train.astype(int))
    print(clfs.best_params_)

    print("================训练集================")
    evalution_model(clfs, x_train, y_train)
    print("================测试集================")
    evalution_model(clfs, x_test, y_test)
    print("===========b_test===================")
    evalution_model(clfs, df_xbtest, df_ybtest)
    # print("================Importance================")
    # plot_importance(clfs, x_train.columns,
    #                 title='feature_importancet', n=30, method="lgb")


def lgb_model(x_train, x_test, y_train, y_test, df_xbtest, df_ybtest, weight_bias=20):
    print("==========LGB===========")
    cout = Counter(y_train)
    tt = cout[0] / cout[1] - weight_bias
    sample_weigh = np.where(y_train == 0, 1, tt)
    dtrain = lgb.Dataset(x_train, list(y_train),
                         categorical_feature="auto",
                         weight=sample_weigh)
    params = {'max_bin': 20,
              'num_leaves': 30,
              'metric': ['l1', 'l2'],
              # 'is_unbalance,': True,
              'learning_rate': 0.01,
              'tree_learner': 'serial',
              'task': 'train',
              'is_training_metric': 'false',
              'min_data_in_leaf': 1,
              'min_sum_hessian_in_leaf': 100,
              'ndcg_eval_at': [1, 3, 5, 10],
              'device': 'cpu',
              'gpu_platform_id': 0,
              'gpu_device_id': 0,
              'feature_fraction': 0.8,
              'max_cat_threshold': 13,
              'force_col_wise': True
              }
    evals_result = {}  # to record eval results for plotting

    clfs = lgb.train(params, train_set=dtrain, num_boost_round=100,
                    valid_sets=[dtrain], valid_names=None,
                    fobj=None, feval=None, init_model=None,
                    categorical_feature='auto',
                    early_stopping_rounds=None, evals_result=evals_result,
                    verbose_eval=10,
                    keep_training_booster=False, callbacks=None,
                     )

    b_pred = clfs.predict(df_xbtest)
    b_pred = np.where(b_pred>0.5 ,1 ,0 )
    pre = round(precision_score(df_ybtest, b_pred, pos_label=1), 2) * 100
    rec = round(recall_score(df_ybtest, b_pred, pos_label=1), 2) * 100
    model_name = "models/lgb_{}_{}.m".format(int(pre), int(rec))
    #joblib.dump(clfs, model_name)


    print("================训练集================")
    evalution_model(clfs, x_train, y_train)
    print("================测试集================")
    evalution_model(clfs, x_test, y_test)
    print("===========b_test===================")
    evalution_model(clfs, df_xbtest, df_ybtest)
    print("================Importance================")
    plot_importance(clfs, x_train.columns,
                    title='feature_importancet', n=30, method="lgb")


def xgb_model(x_train, x_test, y_train, y_test, df_xbtest, df_ybtest):
    print('-------------------XGBOOST-------------------------')
    clfs = XGBClassifier(
        max_depth=7,
        min_child_weight=1,
        learning_rate=0.01,
        n_estimators=20,
        objective='binary:logistic',
        gamma=0,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=0,
        scale_pos_weight=1,
        seed=1,
        missing=None,
        use_label_encoder=False,
        random_state=5)

    cout = Counter(y_train)
    tt = cout[0] / cout[1] - 25
    sample_weigh = np.where(y_train == 0, 1, tt)
    param_grid = {
        "learning_rate": [0.1],
        "n_estimators": range(23, 25),
        "subsample": [0.85],  # 取多少样本，放过拟合
        "scale_pos_weight": [i for i in np.linspace(tt, tt+1, 1)],  # 类似class_weight
        # "max_features": range(7, 8),
        "min_child_weight": range(6, 7),
        "max_depth": range(5, 8),
    }
    dt_score = make_scorer(f1_score, pos_label=1)
    clfs = GridSearchCV(estimator=clfs,
                        param_grid=param_grid,
                        scoring=dt_score,
                        cv=5,
                        # n_jobs=-1
                        )

    clfs = clfs.fit(
        x_train, y_train.astype(int),
        eval_metric='auc', verbose=False,
        eval_set=[(x_test, y_test.astype(int))],
        early_stopping_rounds=100,
    )

    b_pred = clfs.predict(df_xbtest)
    pre = round(precision_score(df_ybtest, b_pred, pos_label=1), 2) * 100
    rec = round(recall_score(df_ybtest, b_pred, pos_label=1), 2) * 100
    model_name = "models/xgb_{}_{}.m".format(int(pre), int(rec))
    #joblib.dump(clfs, model_name)
    print(clfs.best_params_)

    print("================训练集================")
    evalution_model(clfs, x_train, y_train)
    print("================测试集==============")
    evalution_model(clfs, x_test, y_test)
    print("===========b_test===================")
    evalution_model(clfs, df_xbtest,
                    df_ybtest)

    print("================Importance================")
    plot_importance(clfs, x_train.columns,
                    title='feature_importancet', n=30, method=0)


def cat_boost_model(x_train,x_test,y_train,y_test,df_xbtest,df_ybtest,cat_features=[]):
    # cat_features = []
    # for i in range(x_train.shape[1]):
    #     if len(pd.value_counts(x_train.iloc[:, i]).index) <= 80:
    #         cat_features.append(i)
    print('-------------------CATBOOST-------------------------')
    cout = Counter(y_train)
    tt = cout[0] / cout[1] - 25
    sample_weigh = np.where(y_train == 0, 1, tt)
    clfs = CatBoostClassifier(
                              learning_rate=0.01, depth=9, l2_leaf_reg=0.1,
                              loss_function='Logloss', thread_count=8
                              )

    param_grid = {
        "learning_rate": [0.1],
        "n_estimators": range(21, 24),
        # "loss_function": ['MultiClass',
        #                   # 'Logloss'
        #                   ],  # 取多少样本，放过拟合
    #     # "iterations": [40, 21],  # 类似class_weight
        'class_weights': [[1, i] for i in np.linspace(tt, tt+1, 1)],
        "depth": range(6, 9),
    #     # "class_weights" :[1, 7],
    }
    
    dt_score = make_scorer(f1_score, pos_label=1)
    clfs = GridSearchCV(estimator=clfs,
                        param_grid=param_grid,
                        scoring=dt_score,
                        cv=5,
                        n_jobs=-1)

    clfs = clfs.fit(x_train, y_train.astype(int),
                    logging_level=None,
                    plot=False,
                    # sample_weight=sample_weigh,
                    cat_features=cat_features,
                    verbose=None)
    print(clfs.best_params_)
    # print(clfs.best_params_)
    b_pred = clfs.predict(df_xbtest)
    pre = round(precision_score(df_ybtest, b_pred, pos_label=1), 2) * 100
    rec = round(recall_score(df_ybtest, b_pred, pos_label=1), 2) * 100
    model_name = "models/catboost_{}_{}.m".format(int(pre), int(rec))
    #joblib.dump(clfs, model_name)

    print("================训练集================")
    evalution_model(clfs, x_train, y_train)
    print("================测试集==============")
    evalution_model(clfs, x_test, y_test)
    print("===========b_test===================")
    evalution_model(clfs, df_xbtest,
                    df_ybtest)
    plot_importance(clfs, x_train.columns,
                    title='feature_importancet', n=30, method=0)


def gauss_navie_bayes(x_train,x_test,y_train,y_test,df_xbtest,df_ybtest):
    print('-------------------gauss_navie_bayes-------------------------')
    cout = Counter(y_train)
    tt = cout[0] / cout[1]  - 20 +0.5
    sample_weigh = np.where(y_train == 0, 1, tt)
    clfs = GaussianNB()

    clfs = clfs.fit(x_train, y_train.astype(int),
                    sample_weight=sample_weigh)
    # print(clfs.best_params_)
    # print(clfs.best_params_)
    b_pred = clfs.predict(df_xbtest)
    pre = round(precision_score(df_ybtest, b_pred, pos_label=1), 2) * 100
    rec = round(recall_score(df_ybtest, b_pred, pos_label=1), 2) * 100
    model_name = "models/GaussianNB_{}_{}.m".format(int(pre), int(rec))
    #joblib.dump(clfs, model_name)

    print("================训练集================")
    evalution_model(clfs, x_train, y_train)
    print("================测试集==============")
    evalution_model(clfs, x_test, y_test)
    print("===========b_test===================")
    evalution_model(clfs, df_xbtest,
                    df_ybtest)


import os
from sklearn.feature_selection import RFECV
def majorvote_rfecv_by_models(x,df_btest,label="new_new_isSuscess",unuse_column=["student_no", "teacherId"],scoring=precision_score,models=[],filename="select_features.csv"):
    save_dir = 'model_saved/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    load_path = save_dir
    filenames = os.listdir(load_path)
    unuse_columns = [label] + unuse_column
    path = "model_saved/" + filename

    if filename not in filenames:
        y = x[label]
        columns = pd.DataFrame(x.drop(unuse_columns, axis=1).columns).rename(columns={0: "features"})
        i = 1
        for clf in models:
            selector = RFECV(estimator=clf, step=1, cv=5,scoring=scoring)
            selector = selector.fit(x.drop(unuse_columns, axis=1), y)
            neme = "model_" + str(i)
            sl = pd.DataFrame(selector.support_).rename(columns={0: neme})
            columns = pd.concat([columns, sl], axis=1)
            i = i + 1

        for colm in columns.columns[1:]:
            columns[colm] = np.where(columns[colm] == True, 1, 0)
        # columns = np.where(columns==True,1,0)

        sum = 0
        for j in range(len(models)):
            sum = sum + columns.iloc[:, j + 1]
        columns["sum"] = sum

        columns_select = columns[columns["sum"] > len(models) / 2]
        columns_select.to_csv(path, encoding='utf-8', index=None)

    else:
        columns_select = pd.read_csv(path,encoding='utf-8')

    select_features = list(columns_select["features"])
    x_select = pd.concat(
        [pd.DataFrame(x[unuse_columns]).reset_index(drop=True), x[select_features].reset_index(drop=True)],
        axis=1)


    df_btest = pd.concat([df_btest[unuse_columns].reset_index(drop=True),
                          df_btest[select_features].reset_index(drop=True)], axis=1)
    return x_select,df_btest,select_features


from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
# from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator


def major_vote_model(x_train, x_test, y_train, y_test, df_btest, model_weight=[],boundary=0.4):
    print("========majorvote=====")
    cout = Counter(y_train)
    tt = cout[0] / cout[1]  - 20
    sample_weigh = np.where(y_train == 0, 1, tt)
    clf1=RandomForestClassifier(n_estimators=20, max_depth=10, max_features=11, random_state=5,criterion='gini'
                                ,class_weight={1:3.2000000000000002})
    clf2=GradientBoostingClassifier(loss="deviance", learning_rate=0.02,
                                      n_estimators=20, subsample=0.5,max_features=7,
                                      criterion="friedman_mse",warm_start=True,
                                      min_samples_split=2, min_samples_leaf=1,
                                      max_depth=5, random_state=5)
    clf3=XGBClassifier(
              max_depth=7,
              min_child_weight=1,
              learning_rate=0.1,
              n_estimators=21,
              silent=True,
              objective='binary:logistic',
              gamma=0,
              max_delta_step=0,
              subsample=1,
              colsample_bytree=1,
              colsample_bylevel=1,
              reg_alpha=0,
              reg_lambda=0,
              scale_pos_weight=3.687,
              seed=1,
              missing=None,
              random_state=5)
    #lgb

    clf4=CatBoostClassifier(learning_rate=0.01, depth=9, l2_leaf_reg=0.1, loss_function='Logloss',class_weights=[1, 2.8],
                            thread_count=24, random_state=5)
    from sklearn.model_selection import cross_val_score
    clf = MajorityVoteClassifier(classifiers=[clf1,clf2,clf3,clf4],
                                 weights=model_weight,
                                 vote="probability",
                                 boundary = boundary
                                 )
    dt_score = make_scorer(f1_score, pos_label=1)
    clfs = clf.fit(x_train,y_train.astype(int))
    # clfs = cross_val_score(estimator=clf,X=x_train,y=y_train,cv=5,scoring=dt_score)
    y_pred = pd.DataFrame(clfs.predict_proba(x_train))
    y_pred.to_csv('model_saved/pred.csv',encoding='utf-8')

    print("================训练集================")
    evalution_model(clfs, x_train, y_train.astype(int))
    print("================测试集================")
    evalution_model(clfs, x_test, y_test.astype(int))
    print("===========b_test===================")
    evalution_model(clfs, df_xbtest,
                    df_ybtest.astype(int))


def svm_model(x_train, x_test, y_train, y_test, df_btest):
    # cat_features = []
    # for i in range(x_train.shape[1]):
    #     if len(pd.value_counts(x_train.iloc[:, i]).index) <= 80:
    #         cat_features.append(i)
    print('-------------------SVM-------------------------')
    cout = Counter(y_train)
    tt = cout[0] / cout[1]  - 20
    sample_weigh = np.where(y_train == 0, 1, tt)
    clfs = SVC(C=1.0, kernel="rbf", gamma="auto",)
    param_grid = {
                  "C":[10],
                  "kernel":["rbf"],
                  "gamma":[0.1]
                  }
    dt_score = make_scorer(f1_score, pos_label=1)
    clfs = GridSearchCV(estimator=clfs,
                        param_grid=param_grid,
                        scoring=dt_score,
                        cv=5,
                        n_jobs=-1)

    clfs = clfs.fit(x_train, y_train.astype(int),
                    sample_weight=sample_weigh)
    # print(clfs.best_params_)

    print("================训练集================")
    evalution_model(clfs, x_train, y_train)
    print("================测试集==============")
    evalution_model(clfs, x_test, y_test)
    print("===========b_test===================")


import numpy as np
from sklearn.model_selection import KFold
import numpy as np
from scipy.special import expit
import sys


class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_output : int
      Number of output units, should be equal to the
      number of unique class labels.

    n_features : int
      Number of features (dimensions) in the target dataset.
      Should be equal to the number of columns in the X array.

    n_hidden : int (default: 30)
      Number of hidden units.

    l1 : float (default: 0.0)
      Lambda value for L1-regularization.
      No regularization if l1=0.0 (default)

    l2 : float (default: 0.0)
      Lambda value for L2-regularization.
      No regularization if l2=0.0 (default)

    epochs : int (default: 500)
      Number of passes over the training set.

    eta : float (default: 0.001)
      Learning rate.

    alpha : float (default: 0.0)
      Momentum constant. Factor multiplied with the
      gradient of the previous epoch t-1 to improve
      learning speed
      w(t) := w(t) - (grad(t) + alpha*grad(t-1))

    decrease_const : float (default: 0.0)
      Decrease constant. Shrinks the learning rate
      after each epoch via eta / (1 + epoch*decrease_const)

    shuffle : bool (default: False)
      Shuffles training data every epoch if True to prevent circles.

    minibatches : int (default: 1)
      Divides training data into k minibatches for efficiency.
      Normal gradient descent learning if k=1 (default).

    random_state : int (default: None)
      Set random state for shuffling and initializing the weights.

    Attributes
    -----------
    cost_ : list
      Sum of squared errors after each epoch.

    """

    def __init__(self, n_output, n_features, n_hidden=30,
                 l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0, shuffle=True,
                 minibatches=1, random_state=None):

        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y, k):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y : array, shape = [n_samples]
            Target values.

        Returns
        -----------
        onehot : array, shape = (n_labels, n_samples)

        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        """Initialize weights with small random numbers."""
        w1 = np.random.uniform(-1.0, 1.0,
                               size=self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0,
                               size=self.n_output * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)

        Uses scipy.special.expit to avoid overflow
        error for very small input values z.

        """
        # return 1.0 / (1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self, z):
        """Compute gradient of the logistic function"""
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _add_bias_unit(self, X, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0"""
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def _feedforward(self, X, w1, w2):
        """Compute feedforward step

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
          Input layer with original features.

        w1 : array, shape = [n_hidden_units, n_features]
          Weight matrix for input layer -> hidden layer.

        w2 : array, shape = [n_output_units, n_hidden_units]
          Weight matrix for hidden layer -> output layer.

        Returns
        ----------
        a1 : array, shape = [n_samples, n_features+1]
          Input values with bias unit.

        z2 : array, shape = [n_hidden, n_samples]
          Net input of hidden layer.

        a2 : array, shape = [n_hidden+1, n_samples]
          Activation of hidden layer.

        z3 : array, shape = [n_output_units, n_samples]
          Net input of output layer.

        a3 : array, shape = [n_output_units, n_samples]
          Activation of output layer.

        """
        a1 = self._add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        """Compute L2-regularization cost"""
        return (lambda_ / 2.0) * (
                np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

    def _L1_reg(self, lambda_, w1, w2):
        """Compute L1-regularization cost"""
        return (lambda_ / 2.0) * (
                np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        """Compute cost function.

        y_enc : array, shape = (n_labels, n_samples)
          one-hot encoded class labels.

        output : array, shape = [n_output_units, n_samples]
          Activation of the output layer (feedforward)

        w1 : array, shape = [n_hidden_units, n_features]
          Weight matrix for input layer -> hidden layer.

        w2 : array, shape = [n_output_units, n_hidden_units]
          Weight matrix for hidden layer -> output layer.

        Returns
        ---------
        cost : float
          Regularized cost.

        """
        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        """ Compute gradient step using backpropagation.

        Parameters
        ------------
        a1 : array, shape = [n_samples, n_features+1]
          Input values with bias unit.

        a2 : array, shape = [n_hidden+1, n_samples]
          Activation of hidden layer.

        a3 : array, shape = [n_output_units, n_samples]
          Activation of output layer.

        z2 : array, shape = [n_hidden, n_samples]
          Net input of hidden layer.

        y_enc : array, shape = (n_labels, n_samples)
          one-hot encoded class labels.

        w1 : array, shape = [n_hidden_units, n_features]
          Weight matrix for input layer -> hidden layer.

        w2 : array, shape = [n_output_units, n_hidden_units]
          Weight matrix for hidden layer -> output layer.

        Returns
        ---------

        grad1 : array, shape = [n_hidden_units, n_features]
          Gradient of the weight matrix w1.

        grad2 : array, shape = [n_output_units, n_hidden_units]
            Gradient of the weight matrix w2.

        """
        # backpropagation
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # regularize
        grad1[:, 1:] += (w1[:, 1:] * (self.l1 + self.l2))
        grad2[:, 1:] += (w2[:, 1:] * (self.l1 + self.l2))

        return grad1, grad2

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
          Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_samples]
          Predicted class labels.

        """
        if len(X.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')

        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, X, y, print_progress=False):
        """ Learn weights from training data.

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
          Input layer with original features.

        y : array, shape = [n_samples]
          Target class labels.

        print_progress : bool (default: False)
          Prints progress as the number of epochs
          to stderr.

        Returns:
        ----------
        self

        """
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):

            # adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx],
                                      output=a3,
                                      w1=self.w1,
                                      w2=self.w2)
                self.cost_.append(cost)

                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
                                                  a3=a3, z2=z2,
                                                  y_enc=y_enc[:, idx],
                                                  w1=self.w1,
                                                  w2=self.w2)

                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self


class MLPGradientCheck(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_output : int
      Number of output units, should be equal to the
      number of unique class labels.

    n_features : int
      Number of features (dimensions) in the target dataset.
      Should be equal to the number of columns in the X array.

    n_hidden : int (default: 30)
      Number of hidden units.

    l1 : float (default: 0.0)
      Lambda value for L1-regularization.
      No regularization if l1=0.0 (default)

    l2 : float (default: 0.0)
      Lambda value for L2-regularization.
      No regularization if l2=0.0 (default)

    epochs : int (default: 500)
      Number of passes over the training set.

    eta : float (default: 0.001)
      Learning rate.

    alpha : float (default: 0.0)
      Momentum constant. Factor multiplied with the
      gradient of the previous epoch t-1 to improve
      learning speed
      w(t) := w(t) - (grad(t) + alpha*grad(t-1))

    decrease_const : float (default: 0.0)
      Decrease constant. Shrinks the learning rate
      after each epoch via eta / (1 + epoch*decrease_const)

    shuffle : bool (default: False)
      Shuffles training data every epoch if True to prevent circles.

    minibatches : int (default: 1)
      Divides training data into k minibatches for efficiency.
      Normal gradient descent learning if k=1 (default).

    random_state : int (default: None)
      Set random state for shuffling and initializing the weights.

    Attributes
    -----------
    cost_ : list
      Sum of squared errors after each epoch.

    """

    def __init__(self, n_output, n_features, n_hidden=30,
                 l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0, shuffle=True,
                 minibatches=1, random_state=None, class_weight=[1, 2]):

        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches
        self.class_weight = class_weight

    def _encode_labels(self, y, k):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y : array, shape = [n_samples]
            Target values.

        Returns
        -----------
        onehot : array, shape = (n_labels, n_samples)

        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        """Initialize weights with small random numbers."""
        w1 = np.random.uniform(-1.0, 1.0,
                               size=self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0,
                               size=self.n_output * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)

        Uses scipy.special.expit to avoid overflow
        error for very small input values z.

        """
        # return 1.0 / (1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self, z):
        """Compute gradient of the logistic function"""
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _add_bias_unit(self, X, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0"""
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def _feedforward(self, X, w1, w2):
        """Compute feedforward step

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
          Input layer with original features.

        w1 : array, shape = [n_hidden_units, n_features]
          Weight matrix for input layer -> hidden layer.

        w2 : array, shape = [n_output_units, n_hidden_units]
          Weight matrix for hidden layer -> output layer.

        Returns
        ----------
        a1 : array, shape = [n_samples, n_features+1]
          Input values with bias unit.

        z2 : array, shape = [n_hidden, n_samples]
          Net input of hidden layer.

        a2 : array, shape = [n_hidden+1, n_samples]
          Activation of hidden layer.

        z3 : array, shape = [n_output_units, n_samples]
          Net input of output layer.

        a3 : array, shape = [n_output_units, n_samples]
          Activation of output layer.

        """
        a1 = self._add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        """Compute L2-regularization cost"""
        return (lambda_ / 2.0) * (
                np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

    def _L1_reg(self, lambda_, w1, w2):
        """Compute L1-regularization cost"""
        return (lambda_ / 2.0) * (
                np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        """Compute cost function.

        y_enc : array, shape = (n_labels, n_samples)
          one-hot encoded class labels.

        output : array, shape = [n_output_units, n_samples]
          Activation of the output layer (feedforward)

        w1 : array, shape = [n_hidden_units, n_features]
          Weight matrix for input layer -> hidden layer.

        w2 : array, shape = [n_output_units, n_hidden_units]
          Weight matrix for hidden layer -> output layer.

        Returns
        ---------
        cost : float
          Regularized cost.

        """
        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        """ Compute gradient step using backpropagation.

        Parameters
        ------------
        a1 : array, shape = [n_samples, n_features+1]
          Input values with bias unit.

        a2 : array, shape = [n_hidden+1, n_samples]
          Activation of hidden layer.

        a3 : array, shape = [n_output_units, n_samples]
          Activation of output layer.

        z2 : array, shape = [n_hidden, n_samples]
          Net input of hidden layer.

        y_enc : array, shape = (n_labels, n_samples)
          one-hot encoded class labels.

        w1 : array, shape = [n_hidden_units, n_features]
          Weight matrix for input layer -> hidden layer.

        w2 : array, shape = [n_output_units, n_hidden_units]
          Weight matrix for hidden layer -> output layer.

        Returns
        ---------

        grad1 : array, shape = [n_hidden_units, n_features]
          Gradient of the weight matrix w1.

        grad2 : array, shape = [n_output_units, n_hidden_units]
            Gradient of the weight matrix w2.

        """
        # backpropagation
        sigma3 = a3 - y_enc
        # sigma3 = a3 * ((self.class_weight-1)*y_enc +1) - self.class_weight*y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # regularize
        grad1[:, 1:] += (w1[:, 1:] * (self.l1 + self.l2))
        grad2[:, 1:] += (w2[:, 1:] * (self.l1 + self.l2))

        return grad1, grad2

    def _gradient_checking(self, X, y_enc, w1, w2, epsilon, grad1, grad2):
        """ Apply gradient checking (for debugging only)

        Returns
        ---------
        relative_error : float
          Relative error between the numerically
          approximated gradients and the backpropagated gradients.

        """
        num_grad1 = np.zeros(np.shape(w1))
        epsilon_ary1 = np.zeros(np.shape(w1))
        for i in range(w1.shape[0]):
            for j in range(w1.shape[1]):
                epsilon_ary1[i, j] = epsilon
                a1, z2, a2, z3, a3 = self._feedforward(X, w1 - epsilon_ary1, w2)
                cost1 = self._get_cost(y_enc, a3, w1 - epsilon_ary1, w2)
                a1, z2, a2, z3, a3 = self._feedforward(X, w1 + epsilon_ary1, w2)
                cost2 = self._get_cost(y_enc, a3, w1 + epsilon_ary1, w2)
                num_grad1[i, j] = (cost2 - cost1) / (2 * epsilon)
                epsilon_ary1[i, j] = 0

        num_grad2 = np.zeros(np.shape(w2))
        epsilon_ary2 = np.zeros(np.shape(w2))
        for i in range(w2.shape[0]):
            for j in range(w2.shape[1]):
                epsilon_ary2[i, j] = epsilon
                a1, z2, a2, z3, a3 = self._feedforward(X, w1, w2 - epsilon_ary2)
                cost1 = self._get_cost(y_enc, a3, w1, w2 - epsilon_ary2)
                a1, z2, a2, z3, a3 = self._feedforward(X, w1, w2 + epsilon_ary2)
                cost2 = self._get_cost(y_enc, a3, w1, w2 + epsilon_ary2)
                num_grad2[i, j] = (cost2 - cost1) / (2 * epsilon)
                epsilon_ary2[i, j] = 0

        num_grad = np.hstack((num_grad1.flatten(), num_grad2.flatten()))
        grad = np.hstack((grad1.flatten(), grad2.flatten()))
        norm1 = np.linalg.norm(num_grad - grad)
        norm2 = np.linalg.norm(num_grad)
        norm3 = np.linalg.norm(grad)
        relative_error = norm1 / (norm2 + norm3)
        return relative_error

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
          Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_samples]
          Predicted class labels.

        """
        if len(X.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')

        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, X, y, print_progress=False):
        """ Learn weights from training data.

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
          Input layer with original features.

        y : array, shape = [n_samples]
          Target class labels.

        print_progress : bool (default: False)
          Prints progress as the number of epochs
          to stderr.

        Returns:
        ----------
        self

        """
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):

            # adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:

                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx],
                                      output=a3,
                                      w1=self.w1,
                                      w2=self.w2)
                self.cost_.append(cost)


                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
                                                  a3=a3, z2=z2,
                                                  y_enc=y_enc[:, idx],
                                                  w1=self.w1,
                                                  w2=self.w2)

                # print('loss:' + str(cost)+'  grad1:'+str(grad1))
                ## start gradient checking
                grad_diff = self._gradient_checking(X=X[idx],
                                                    y_enc=y_enc[:, idx],
                                                    w1=self.w1, w2=self.w2,
                                                    epsilon=1e-5,
                                                    grad1=grad1, grad2=grad2)

                if grad_diff <= 1e-7:
                    print('Ok: %s' % grad_diff)
                elif grad_diff <= 1e-4:
                    print('Warning: %s' % grad_diff)
                else:
                    print('PROBLEM: %s' % grad_diff)

                # update weights; [alpha * delta_w_prev] for momentum learning
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self


def MLPGradientCheck_model(x_train, x_test, y_train, y_test, df_xbtest, df_ybtest):
    print('-------------------MLPGradientCheck_model-------------------------')
    cout = Counter(y_train)
    tt = cout[0] / cout[1] - 23
    sample_weigh = np.where(y_train == 0, 1, tt)
    clfs = MLPGradientCheck(n_output=2,
                            n_features=x_train.shape[1],
                            n_hidden=5,
                            l2=0.0,
                            l1=0.01,
                            epochs=200,
                            eta=0.001,
                            alpha=0.001,
                            decrease_const=0.001,
                            minibatches=1,
                            random_state=1)

    clfs = clfs.fit(x_train, y_train.astype(int), print_progress=True)
    b_pred = clfs.predict(df_xbtest)
    pre = round(precision_score(df_ybtest, b_pred, pos_label=1), 2) * 100
    rec = round(recall_score(df_ybtest, b_pred, pos_label=1), 2) * 100
    model_name = "models/catboost_{}_{}.m".format(int(pre), int(rec))
    #joblib.dump(clfs, model_name)

    print("================训练集================")
    evalution_model(clfs, x_train, y_train)
    print("================测试集==============")
    evalution_model(clfs, x_test, y_test)
    print("===========b_test===================")
    evalution_model(clfs, df_xbtest,
                    df_ybtest)


def gbdt_plus_lr(x_train, x_test, y_train, y_test, df_xbtest, df_ybtest, numeric_features=[]):
    train_cat = pd.DataFrame()
    test_cat = pd.DataFrame()
    btest_cat = pd.DataFrame()
    if len(numeric_features) > 0:
        x_train_cat = x_train.drop(numeric_features, axis=1)
        x_test_cat = x_train.drop(numeric_features, axis=1)
        x_btest_cat = x_train.drop(numeric_features, axis=1)

        #  onehot编码
        enc = OneHotEncoder()
        enc.fit(x_train_cat)
        train_cat = np.array(enc.transform(x_train_cat).toarray())
        test_cat = np.array(enc.transform(x_test_cat).toarray())
        btest_cat = np.array(enc.transform(x_btest_cat).toarray())

        x_train = x_train[numeric_features]
        x_test = x_test[numeric_features]
        df_xbtest = df_xbtest[numeric_features]

    # create dataset for lightgbm
    print("==========LGB+LR===========")
    cout = Counter(y_train)
    tt = cout[0] / cout[1] - 20
    sample_weigh = np.where(y_train == 0, 1, tt)
    dtrain = lgb.Dataset(x_train, list(y_train),
                         categorical_feature="auto",
                         weight=sample_weigh)
    params = {'max_bin': 20,
              'num_leaves': 30,
              'metric': ['l1', 'l2'],
              # 'is_unbalance,': True,
              'learning_rate': 0.01,
              'tree_learner': 'serial',
              'task': 'train',
              'is_training_metric': 'false',
              'min_data_in_leaf': 1,
              'min_sum_hessian_in_leaf': 100,
              'ndcg_eval_at': [1, 3, 5, 10],
              'device': 'cpu',
              'gpu_platform_id': 0,
              'gpu_device_id': 0,
              'feature_fraction': 0.8,
              'max_cat_threshold': 13,
              'force_col_wise': True
              }
    evals_result = {}  # to record eval results for plotting

    clfs = lgb.train(params, train_set=dtrain, num_boost_round=100,
                     valid_sets=[dtrain], valid_names=None,
                     fobj=None, feval=None, init_model=None,
                     categorical_feature='auto',
                     early_stopping_rounds=None, evals_result=evals_result,
                     verbose_eval=10,
                     keep_training_booster=False, callbacks=None,
                     )
    print("================训练集================")
    evalution_model(clfs, x_train, y_train)
    print("================测试集================")
    evalution_model(clfs, x_test, y_test)
    print("===========b_test===================")
    evalution_model(clfs, df_xbtest, df_ybtest)
    #  查找每棵树叶子节点
    y_pred_train = clfs.predict(x_train, pred_leaf=True)
    y_pred_test = clfs.predict(x_test, pred_leaf=True)
    y_pred_btest = clfs.predict(df_xbtest, pred_leaf=True)

    #  onehot编码
    enc = OneHotEncoder()
    enc.fit(y_pred_train)
    train_encode = np.array(enc.transform(y_pred_train).toarray())
    test_encode = np.array(enc.transform(y_pred_test).toarray())
    btest_encode = np.array(enc.transform(y_pred_btest).toarray())
    print(train_encode.shape)
    print(train_cat.shape)
    # 合并
    train_encodes = pd.concat([pd.DataFrame(train_encode), pd.DataFrame(train_cat)], axis=1).astype(np.float32)
    test_encodes = pd.concat([pd.DataFrame(test_encode), pd.DataFrame(test_cat)], axis=1).astype(np.float32)
    btest_encodes = pd.concat([pd.DataFrame(btest_encode), pd.DataFrame(btest_cat)], axis=1).astype(np.float32)

    # LR
    tt = cout[0] / cout[1] - 28
    sample_weigh = np.where(y_train == 0, 1, tt)
    lr = LogisticRegression(penalty='l2', C=0.05)
    lr.fit(train_encodes, y_train, sample_weight=sample_weigh)

    print("================训练集================")
    evalution_model(lr, train_encodes, y_train)
    print("================测试集==============")
    evalution_model(lr, test_encodes, y_test)
    print("===========b_test===================")
    evalution_model(lr, btest_encodes, df_ybtest)


def gcforest(x_train, x_test, y_train, y_test, df_xbtest, df_ybtest):
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    df_xbtest = np.array(df_xbtest)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    df_xbtest = np.array(df_xbtest)

    print("==========GCfroset==========")
    cout = Counter(y_train)
    tt = cout[0] / cout[1] - 23
    def get_toy_config():
        config = {}
        ca_config = {}
        ca_config["random_state"] = 0
        ca_config["max_layers"] = 100
        ca_config["early_stopping_rounds"] = 3
        ca_config["n_classes"] = 2
        ca_config["estimators"] = []
        ca_config["estimators"].append(
            {"n_folds": 2, "type": "RandomForestClassifier", "n_estimators": 20, "max_depth": 5, "n_jobs": -1, "class_weight":{0: 1, 1: tt}, 'criterion': 'gini'})
        ca_config["estimators"].append(
            {"n_folds": 2, "type": "ExtraTreesClassifier", "n_estimators": 20, "max_depth": 5, "n_jobs": -1, "class_weight":{0: 1, 1: tt}, 'criterion': 'gini'})
        ca_config["estimators"].append({"n_folds": 2, "type": "XGBClassifier", 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 6, 'n_estimators': 23, 'scale_pos_weight': tt, 'subsample': 0.85})
        config["cascade"] = ca_config
        return config

    # 模型参数
    config = get_toy_config()

    # 模型初始化
    gc = GCForest(config)
    # 模型训练
    gc.fit_transform(x_train, y_train)

    # y_pred = gc.predict(x_test)

    print("================训练集================")
    evalution_model(gc, x_train, y_train)
    print("================测试集================")
    evalution_model(gc, x_test, y_test)
    print("===========b_test===================")
    evalution_model(gc, df_xbtest,
                    df_ybtest)


def gcforest2(x_train, x_test, y_train, y_test, df_xbtest, df_ybtest):
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    df_xbtest = np.array(df_xbtest)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    df_xbtest = np.array(df_xbtest)

    print("==========GCfroset==========")
    cout = Counter(y_train)
    tt = cout[0] / cout[1] - 23


    # 模型初始化
    gc = gcForest(shape_1X=x_train.shape[1], n_mgsRFtree=30, window=5, stride=1, weight=tt, n_layer=6,
                 cascade_test_size=0.2, n_cascadeRF=2, n_cascadeRFtree=101, cascade_layer=10,
                 min_samples_mgs=0.1, min_samples_cascade=0.05, tolerance=0.0, n_jobs=-1)
    # 模型训练
    gc.fit(x_train, y_train)

    # y_pred = gc.predict(x_test)

    print("================训练集================")
    evalution_model(gc, x_train, y_train)
    print("================测试集================")
    evalution_model(gc, x_test, y_test)
    print("===========b_test===================")
    evalution_model(gc, df_xbtest,
                    df_ybtest)


# kf = KFold(5, random_state=0)


def stacking_reg(clf, train_x, train_y, test_x, clf_name, folds, kf, label_split=None):
    train = np.zeros((train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))
    test_pre = np.empty((folds, test_x.shape[0], 1))
    cv_scores = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x, label_split)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        if clf_name in ["rf", "ada", "gb", "et", "lr"]:
            clf.fit(tr_x, tr_y)
            pre = clf.predict(te_x).reshape(-1, 1)
            train[test_index] = pre
            test_pre[i, :] = clf.predict(test_x).reshape(-1, 1)
            cv_scores.append(mean_squared_error(te_y, pre))
        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=te_y, missing=-1)
            params = {'booster': 'gbtree',
                      'eval_metric': 'rmse',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12
                      }
            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round=num_round,evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre = model.predict(test_matrix, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(z, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pre))

        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            params = {
                      'boosting_type': 'gbdt',
                      'objective': 'regression_l2',
                      'metric': 'mse',
                      'min_child_weight': 1.5,
                      'num_leaves': 2**5,
                      'lambda_l2': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'learning_rate': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12,
                      'silent': True,
                      }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre = model.predict(te_x,num_iteration=model.best_iteration).reshape(-1,1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(test_x, num_iteration=model.best_iteration).reshape(-1,1)
                cv_scores.append(mean_squared_error(te_y, pre))
        else:
            raise IOError("Please add new clf.")
        print("%s now score is:" % clf_name, cv_scores)
    test[:] = test_pre.mean(axis=0)
    print("%s_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    return train.reshape(-1, 1), test.reshape(-1, 1)


def stacking_clf(clf, train_x, train_y, test_x, clf_name, folds, kf, label_split=None):

    cout = Counter(train_y)
    tt = cout[0] / cout[1] - 23
    sample_weigh = np.where(train_y == 0, 1, tt)

    clf1 = RandomForestClassifier(n_estimators=24,
                                  max_depth=5,
                                  max_features=5,
                                  random_state=5,
                                  n_jobs=-1
                                  )
    clf2 = GradientBoostingClassifier(loss="deviance", learning_rate=0.01,
                                      n_estimators=20, subsample=1.0,
                                      criterion="friedman_mse",
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      max_depth=5,
                                      random_state=5)

    clf3 = XGBClassifier(max_depth=7,
                         min_child_weight=1,
                         learning_rate=0.01,
                         n_estimators=20,
                         objective='binary:logistic',
                         gamma=0,
                         max_delta_step=0,
                         subsample=1,
                         colsample_bytree=1,
                         colsample_bylevel=1,
                         reg_alpha=0,
                         reg_lambda=0,
                         scale_pos_weight=1,
                         seed=1,
                         missing=None,
                         use_label_encoder=False,
                         random_state=5)

    dtrain = lgb.Dataset(train_x, list(train_y),
                         categorical_feature="auto",
                         weight=sample_weigh)
    params = {'max_bin': 20,
              'num_leaves': 30,
              'metric': ['l1', 'l2'],
              # 'is_unbalance,': True,
              'learning_rate': 0.01,
              'tree_learner': 'serial',
              'task': 'train',
              'is_training_metric': 'false',
              'min_data_in_leaf': 1,
              'min_sum_hessian_in_leaf': 100,
              'ndcg_eval_at': [1, 3, 5, 10],
              'device': 'cpu',
              'gpu_platform_id': 0,
              'gpu_device_id': 0,
              'feature_fraction': 0.8,
              'max_cat_threshold': 13,
              'force_col_wise': True
              }
    evals_result = {}  # to record eval results for plotting

    clf4 = lgb.train(params, train_set=dtrain, num_boost_round=100,
                     valid_sets=[dtrain], valid_names=None,
                     fobj=None, feval=None, init_model=None,
                     categorical_feature='auto',
                     early_stopping_rounds=None, evals_result=evals_result,
                     verbose_eval=10,
                     keep_training_booster=False, callbacks=None,
                     )


    train = np.zeros((train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))
    test_pre = np.empty((folds, test_x.shape[0], 1))
    cv_scores = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x, label_split)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        if clf_name in ["rf", "ada", "gb", "et", "lr"]:
            clf.fit(tr_x, tr_y)
            pre = clf.predict_proba(te_x).reshape(-1, 1)
            train[test_index] = pre
            test_pre[i, :] = clf.predict(test_x).reshape(-1, 1)
            cv_scores.append(mean_squared_error(te_y, pre))
        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=te_y, missing=-1)
            params = {'booster': 'gbtree',
                      'eval_metric': 'rmse',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12
                      }
            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round=num_round,evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre = model.predict(test_matrix, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(z, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pre))

        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            params = {
                      'boosting_type': 'gbdt',
                      'objective': 'regression_l2',
                      'metric': 'mse',
                      'min_child_weight': 1.5,
                      'num_leaves': 2**5,
                      'lambda_l2': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'learning_rate': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12,
                      'silent': True,
                      }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre = model.predict(te_x, num_iteration=model.best_iteration).reshape(-1,1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(test_x, num_iteration=model.best_iteration).reshape(-1,1)
                cv_scores.append(mean_squared_error(te_y, pre))
        else:
            raise IOError("Please add new clf.")
        print("%s now score is:" % clf_name, cv_scores)
    test[:] = test_pre.mean(axis=0)
    print("%s_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    return train.reshape(-1, 1), test.reshape(-1, 1)


def stacking_models(clf, train_x, train_y, test_x, btest_x, clf_name, folds, label_split=None):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    btest_x = np.array(btest_x)

    print("==============start_stacking================")
    kf = KFold(folds, random_state=0, shuffle=True)

    train = np.zeros((train_x.shape[0], 2))
    test = np.zeros((test_x.shape[0], 2))
    test_pre = np.empty((folds, test_x.shape[0], 2))

    btest = np.zeros((btest_x.shape[0], 2))
    btest_pre = np.empty((folds, btest_x.shape[0], 2))
    cv_scores = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x, label_split)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]

        cout = Counter(tr_y)
        tt = cout[0] / cout[1] - 23
        sample_weigh = np.where(tr_y == 0, 1, tt)
        if clf_name in ["rf", "ada", "gb", "et", "lr"]:
            clf.fit(tr_x, tr_y, sample_weight=sample_weigh)
            pre = clf.predict_proba(te_x)
            pre1 = pre[:, 1]
            train[test_index] = pre
            # test
            test_pre[i, :] = clf.predict_proba(test_x)
            cv_scores.append(mean_squared_error(te_y, pre1))
            # b_test
            btest_pre[i, :] = clf.predict_proba(btest_x)
        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1, weight=sample_weigh)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=te_y, missing=-1)
            bz = clf.DMatrix(btest_x, label=te_y, missing=-1)
            params = {'booster': 'gbtree',
                      'eval_metric': 'rmse',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.8,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12
                      }
            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                model = clf.train(params,
                                  train_matrix,
                                  num_boost_round=num_round,
                                  evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre = model.predict(test_matrix, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(test_x, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
                btest_pre[i, :] = model.predict(btest_x, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pre))

        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y, weight=sample_weigh)
            test_matrix = clf.Dataset(te_x, label=te_y)
            params = {
                      'boosting_type': 'gbdt',
                      'objective': 'regression_l2',
                      'metric': 'mse',
                      'min_child_weight': 1.5,
                      'num_leaves': 2**5,
                      'lambda_l2': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'learning_rate': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12,
                      'silent': True,
                      }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix, num_round, valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre = model.predict(te_x, num_iteration=model.best_iteration).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(test_x, num_iteration=model.best_iteration).reshape(-1, 1)
                btest_pre[i, :] = model.predict(btest_x, num_iteration=model.best_iteration).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pre))
        else:
            raise IOError("Please add new clf.")
        print("%s now score is:" % clf_name, cv_scores)
    test[:] = test_pre.mean(axis=0)
    btest[:] = btest_pre.mean(axis=0)
    print("%s_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    return train, test, btest

