import warnings
warnings.filterwarnings('ignore')
from  data_treatment import load_data_yf,data_clean,seperate_label,data_seperate,load_data_new,plot_eda,data_clean,feature_extend,data_clean2,feature_onehot
from models import lr_model,rf_mdoel,gbdt_mdoel,xgb_model,cat_boost_model,lgb_model, MLPGradientCheck_model,lgb_sk_mdoel,gauss_navie_bayes,gbdt_plus_lr,gcforest,gcforest2, adaboost_model, stacking_models
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from model_evalu import evalution_model,plot_importance
import numpy as np
from xgboost import XGBClassifier
import tpot
from tpot import TPOTClassifier
from sklearn.metrics import precision_score,f1_score,recall_score
import warnings
import numpy as np
import pandas as pd
import pymysql
from collections import Counter
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


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    from sklearn.metrics import precision_score, f1_score, recall_score
    warnings.filterwarnings('ignore')
    # 加载数据
    sql = "SELECT * from bidata.trail_pigeon_wdf1"
    df = load_data_new(sql, filename="drumping.csv")

    label_by_contract = "target_is_DD_ACTIVE"
    labels = label_by_contract

    select_columns = [
                      "is_festival_user",
                      "level_use",
                      "is_LAST_2YEAR_DD_ACTIVE",
                      "cafe_tag_is_mop_available",
                      "CAFE20_region_SH",
                      "is_merch_user",
                      "p4week_active",
                      "is_LAST_1YEAR_DD_ACTIVE",
                      "msr_lifestatus",
                      "IS_SR_KIT_USER",
                      "member_monetary",


                      # "CAFE20_levels_9.Welcome 1",  # onehot
                      # "CAFE20_levels_0.Gold Monthly 8+",
                      "citytier",
                      "active_index",
                      "cafe_tag_p6m_food_qty",
                      "total_amt",
                      "DD_rev",
                      "svc_revenue",
                      "DDoffer_rec",
                      "mop_spend",
                      "recency",
                      "food_party_size",
                      "multi_bev",
                      "MC_red_rate",
                      "SR_KIT_NUM",
                      "CAFE20_RECENCY_MERCH",
                      "cafe_tag_p3m_merch_party_size",
                      "CAFE20_VISIT_MERCH",
                      "CAFE20_P1Y_AVG_TRANX_DAY",
                      "CAFE20_VISIT_APP",
                      "CAFE20_AI",
                      "CAFE20_age",
                      "CAFE20_RECENCY_APP",
                      "CAFE20_RECENCY_bev_food",
                      "CAFE20_AMT",
                      "cafe_tag_p3m_food_qty",
                      "rank_preference_food",
                      "p3m_weekday_trans",
                      "CAFE20_MONTHLY_FREQ",
                      "monthly_freq",
                      "cafe_tag_p6m_merch_party_size",
                      "CAFE20_VISIT_bev_food",
                      "max_DD_rev",
                      "p6m_trans",
                      "cafe_tag_p6m_monthly_freq",
                      "DD_end_gap",
                      "DD_launch_gap",
                      "d10_p8week_active",
                      "p6m_amt",
                      "cafe_tag_p3m_merch_qty",
                      "DD_order_num",
                      "MC_end_gap",
                      "p2w_amt",
                      "CAFE20_VISIT_BEV",
                      "CAFE20_RECENCY",
                      "cafe_tag_p3m_monthly_freq",
                      "cafe_tag_p6m_merch_qty",
                      "p3m_weekly_frq",
                      "total_trans",
                      "DD_units",
                      "max_DD_Quantity",
                      "MC_rev",
                      "p6m_weekday_trans",
                      "MC_launch_gap",
                      "MC_units",
                      "cafe_tag_p3m_vist",
                      "MCoffer_red",
                      "CAFE20_VISIT_SRKIT",
                      "p2w_trans",
                      "CAFE20_RECENCY_SRKIT",
                      "max_MC_rev",
                      "CAFE20_P1Y_VISITS_DAY",
                      "max_MC_Quantity",
                        labels,
                      ]
    catfeatures = ["is_festival_user",
                   "level_use",
                   "is_LAST_2YEAR_DD_ACTIVE",
                   "cafe_tag_is_mop_available",
                   "CAFE20_region_SH",
                   "is_merch_user",
                   "p4week_active",
                   "is_LAST_1YEAR_DD_ACTIVE",
                   "msr_lifestatus",
                   "IS_SR_KIT_USER",
                   "member_monetary"]


    #  数据预处理
    df_train, df_btest = data_clean2(df)
    df_train = df_train[select_columns]
    df_btest = df_btest[select_columns]

    for cats in catfeatures:
        df_train[cats] = df_train[cats].astype(int)
        df_btest[cats] = df_btest[cats].astype(int)

    # # 抽样
    # df_train = df_train.sample(n=None, frac=0.1, replace=False, weights=None,
    #                            random_state=0, axis=0)
    # df_btest = df_btest.sample(n=None, frac=0.1, replace=False, weights=None,
    #                            random_state=0, axis=0)


    print('正/负', str(len(df_train[df_train[labels] == 1])) + '/' + str(len(df_train[df_train[labels] == 0])))
    t = len(df_train[df_train[labels] == 0]) / len(df_train[df_train[labels] == 1])
    v = len(df_btest[df_btest[labels] == 0]) / len(df_btest[df_btest[labels] == 1])
    print(t, v)

    # #划分训练测试集
    X_train_tra, X_test_tra, df_btest= data_seperate(df_train, df_btest, size=0.3, label=labels, cri=None,
                                                     undeal_column=None)

    # # 划分label
    print(df_train.columns)

    #  降采样
    # x0 = X_train_tra[X_train_tra[labels] == 0]
    # x1 = X_train_tra[X_train_tra[labels] == 1]
    # x0 = x0.sample(n=None, frac=0.33, replace=False, weights=None, random_state=0, axis=0)
    # X_train_tra = pd.concat([x0, x1], axis=0)

    #  划分label
    x_train, y_train = seperate_label(X_train_tra, label=labels)
    x_test, y_test = seperate_label(X_test_tra, label=labels)
    print("x_train", x_train.shape)


    numeric_features = ["citytier",
                      "active_index",
                      "cafe_tag_p6m_food_qty",
                      "total_amt",
                      "DD_rev",
                      "svc_revenue",
                      "DDoffer_rec",
                      "mop_spend",
                      "recency",
                      "food_party_size",
                      "multi_bev",
                      "MC_red_rate",
                      "SR_KIT_NUM",
                      "CAFE20_RECENCY_MERCH",
                      "cafe_tag_p3m_merch_party_size",
                      "CAFE20_VISIT_MERCH",
                      "CAFE20_P1Y_AVG_TRANX_DAY",
                      "CAFE20_VISIT_APP",
                      "CAFE20_AI",
                      "CAFE20_age",
                      "CAFE20_RECENCY_APP",
                      "CAFE20_RECENCY_bev_food",
                      "CAFE20_AMT",
                      "cafe_tag_p3m_food_qty",
                      "rank_preference_food",
                      "p3m_weekday_trans",
                      "CAFE20_MONTHLY_FREQ",
                      "monthly_freq",
                      "cafe_tag_p6m_merch_party_size",
                      "CAFE20_VISIT_bev_food",
                      "max_DD_rev",
                      "p6m_trans",
                      "cafe_tag_p6m_monthly_freq",
                      "DD_end_gap",
                      "DD_launch_gap",
                      "d10_p8week_active",
                      "p6m_amt",
                      "cafe_tag_p3m_merch_qty",
                      "DD_order_num",
                      "MC_end_gap",
                      "p2w_amt",
                      "CAFE20_VISIT_BEV",
                      "CAFE20_RECENCY",
                      "cafe_tag_p3m_monthly_freq",
                      "cafe_tag_p6m_merch_qty",
                      "p3m_weekly_frq",
                      "total_trans",
                      "DD_units",
                      "max_DD_Quantity",
                      "MC_rev",
                      "p6m_weekday_trans",
                      "MC_launch_gap",
                      "MC_units",
                      "cafe_tag_p3m_vist",
                      "MCoffer_red",
                      "CAFE20_VISIT_SRKIT",
                      "p2w_trans",
                      "CAFE20_RECENCY_SRKIT",
                      "max_MC_rev",
                      "CAFE20_P1Y_VISITS_DAY",
                      "max_MC_Quantity"]
    df_xbtest = df_btest.drop(labels, axis=1)
    df_ybtest = df_btest[labels]
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
    params = {
               'max_bin': 20,
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
    tt = cout[0] / cout[1] - 26
    sample_weigh = np.where(y_train == 0, 1, tt)
    lr = LogisticRegression(penalty='l2', C=0.05)
    lr.fit(train_encodes, y_train, sample_weight=sample_weigh)

    print("================训练集================")
    evalution_model(lr, train_encodes, y_train)
    print("================测试集==============")
    evalution_model(lr, test_encodes, y_test)
    print("===========b_test===================")
    evalution_model(lr, btest_encodes, df_ybtest)


