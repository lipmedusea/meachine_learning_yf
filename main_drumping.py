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

    # #模型训练
    # adaboost模型
    adaboost_model(x_train, x_test, y_train, y_test, df_btest.drop(labels, axis=1), df_btest[labels])

    #  LR模型
    lr_model(x_train, x_test, y_train, y_test, df_btest.drop(labels, axis=1), df_btest[labels])
    # # rf模型
    rf_mdoel(x_train, x_test, y_train, y_test, df_btest.drop(labels, axis=1), df_btest[labels])
    # # #
    # # gbdt模型
    gbdt_mdoel(x_train, x_test, y_train, y_test, df_btest.drop(labels, axis=1), df_btest[labels])

    #  xgb模型
    xgb_model(x_train, x_test, y_train, y_test, df_btest.drop(labels, axis=1), df_btest[labels])

    # lgb模型
    lgb_model(x_train, x_test, y_train, y_test, df_btest.drop(labels, axis=1), df_btest[labels]
              , weight_bias=20)


    # lgb_sk
    # lgb_sk_mdoel(x_train, x_test, y_train, y_test, df_btest.drop(labels,axis=1),df_btest[labels])

    # catboost
    cat_boost_model(x_train, x_test, y_train, y_test,
                    df_btest.drop(labels, axis=1),
                    df_btest[labels],
                    cat_features=catfeatures)

    #  高斯贝叶斯
    # gauss_navie_bayes(x_train, x_test, y_train, y_test, df_btest.drop(labels,axis=1),df_btest[labels])

    #  gbdt+lr
    gbdt_plus_lr(x_train, x_test, y_train, y_test,
                 df_btest.drop(labels, axis=1), df_btest[labels],
                 numeric_features=[]
                 )
    # # gcforest
    gcforest(x_train, x_test, y_train, y_test,
             df_btest.drop(labels, axis=1), df_btest[labels])

    # # gcforest2
    # gcforest2(x_train, x_test, y_train, y_test,
    #          df_btest.drop(labels, axis=1), df_btest[labels])

    # stacking_model
    clf = RandomForestClassifier(n_estimators=23,
                                 max_depth=6,
                                 max_features=9,
                                 random_state=5,
                                 criterion='gini',
                                 n_jobs=-1,
                                 )
    train, test, btest = stacking_models(clf, x_train, y_train, x_test,  df_btest.drop(labels, axis=1), 'rf', folds=5, label_split=None)

    x_train = pd.concat([x_train, pd.DataFrame(train, columns=["stacking_0", "stacking_1"])], axis=1)
    x_test = pd.concat([x_test, pd.DataFrame(test, columns=["stacking_0", "stacking_1"])], axis=1)
    x_btest = pd.concat([df_btest.drop(labels, axis=1), pd.DataFrame(btest, columns=["stacking_0", "stacking_1"])], axis=1)

    lgb_model(x_train, x_test, y_train, y_test, x_btest, df_btest[labels], weight_bias=22)

