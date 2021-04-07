import warnings
warnings.filterwarnings('ignore')
from  data_treatment import load_data_yf,data_clean,seperate_label,data_seperate,load_data_new,plot_eda,data_clean,feature_extend,data_clean2,feature_onehot
from models import lr_model,rf_mdoel,gbdt_mdoel,xgb_model,cat_boost_model,lgb_model, MLPGradientCheck_model,lgb_sk_mdoel,gauss_navie_bayes,gbdt_plus_lr,gcforest,gcforest2, adaboost_model
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
    df = load_data_new(sql, filename="df_20190226.csv")

    label_by_contract = "is_pigeon"
    labels = label_by_contract

    select_columns = [
        # "sale_id",
        # "teacher_id",
        "know_origin",
        "grade_subject",
        "student_city_class",
        "student_province",
        "grade_rank",
        "class_rank_fillna",
        "student_province_byphone",
        "subject_ids",
        "student_grade_lpo",
        "school_background",
        "is_first_trail",
        "class_background_label",
        'student_grade',
        # "exam_year",
        "coil_in",

        "is_pigeon",

        "score_mean",
        "self_evaluation_length",
        "score_min",
        "learning_target_lenght",
        "score_max",

        "order_month",
        "order_day",
        "order_hour",
        # "lesson_month",
        # "lesson_day",
        # "lesson_hour",

        "sale_pigeon_rate",
        "lm_sale_pigeon_rate",
        "l3m_sale_pigeon_rate",
        "his_sale_pigeon_rate",
        "sale_pigeon_counts",
        "sale_total_counts",
        "lm_sale_pigeon_counts",
        "lm_sale_total_counts",
        "l3m_sale_pigeon_counts",
        "l3m_sale_total_counts",
        "his_sale_pigeon_counts",
        "his_sale_total_counts",

        "high_intention_counts",
        "low_intention_counts",
        "lost_connection_counts",
        "is_have_experience_lesson",
        "commu_counts",
        "high_intenetion_rate",
        "low_intenetion_rate",
        "lost_intenetion_rate",

        # "cnn_prob",
        # "cnn_dens1",
        # "voice_feature"

    ]
    cat_features = [
        # "sale_id",
        # "teacher_id",
        "know_origin",
        "grade_subject",
        "student_city_class",
        "student_province",
        "grade_rank",
        "class_rank_fillna",
        "student_province_byphone",
        "subject_ids",
        "student_grade_lpo",
        "school_background",
        "is_first_trail",
        "class_background_label",
        'student_grade',
        # "exam_year",
        "coil_in"]

    #  数据预处理
    df_train, df_btest = data_clean2(df, min_date="2018-08-21", mid_date="2018-12-03", max_date="2018-12-18", label=labels)

    for f in cat_features:
        df_train[f] = df_train[f].astype(int)
        df_btest[f] = df_btest[f].astype(int)
    df_train = df_train[select_columns]
    df_btest = df_btest[select_columns]
    # 抽样
    df_train = df_train.sample(n=None, frac=0.1, replace=False, weights=None,
                               random_state=0, axis=0)
    df_btest = df_btest.sample(n=None, frac=0.1, replace=False, weights=None,
                               random_state=0, axis=0)


    print('正/负', str(len(df_train[df_train[labels] == 1])) + '/' + str(len(df_train[df_train[labels] == 0])))
    t = len(df_train[df_train[labels] == 0]) / len(df_train[df_train[labels] == 1])
    v = len(df_btest[df_btest[labels] == 0]) / len(df_btest[df_btest[labels] == 1])
    print(t, v)

    # #划分训练测试集
    X_train_tra, X_test_tra, df_btest= data_seperate(df_train, df_btest, size=0.3, label="is_pigeon", cri=None,
                                                     undeal_column=cat_features)

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

    #  LR模型
    # lr_model(x_train, x_test, y_train, y_test, df_btest.drop("is_pigeon", axis=1), df_btest["is_pigeon"])
    # # rf模型
    rf_mdoel(x_train, x_test, y_train, y_test, df_btest.drop("is_pigeon", axis=1), df_btest["is_pigeon"])
    # # #
    # # gbdt模型
    # gbdt_mdoel(x_train, x_test, y_train, y_test, df_btest.drop("is_pigeon",axis=1),df_btest["is_pigeon"])

    #  xgb模型
    xgb_model(x_train, x_test, y_train, y_test, df_btest.drop("is_pigeon", axis=1), df_btest["is_pigeon"])

    # lgb模型
    lgb_model(x_train, x_test, y_train, y_test, df_btest.drop("is_pigeon", axis=1), df_btest["is_pigeon"])


    # lgb_sk
    # lgb_sk_mdoel(x_train, x_test, y_train, y_test, df_btest.drop("is_pigeon",axis=1),df_btest["is_pigeon"])

    # catboost
    cat_boost_model(x_train, x_test, y_train, y_test, df_btest.drop("is_pigeon",axis=1),df_btest["is_pigeon"])

    #  高斯贝叶斯
    # gauss_navie_bayes(x_train, x_test, y_train, y_test, df_btest.drop("is_pigeon",axis=1),df_btest["is_pigeon"])

    #  gbdt+lr
    gbdt_plus_lr(x_train, x_test, y_train, y_test,
                 df_btest.drop("is_pigeon", axis=1), df_btest["is_pigeon"],
                 numeric_features=["score_mean",
                                   "self_evaluation_length",
                                   "score_min",
                                   "learning_target_lenght",
                                   "score_max",
                                   "order_month",
                                   "order_day",
                                   "order_hour",
                                   # "lesson_month",
                                   # "lesson_day",
                                   # "lesson_hour",
                                   "sale_pigeon_rate",
                                   "lm_sale_pigeon_rate",
                                   "l3m_sale_pigeon_rate",
                                   "his_sale_pigeon_rate",
                                   "sale_pigeon_counts",
                                   "sale_total_counts",
                                   "lm_sale_pigeon_counts",
                                   "lm_sale_total_counts",
                                   "l3m_sale_pigeon_counts",
                                   "l3m_sale_total_counts",
                                   "his_sale_pigeon_counts",
                                   "his_sale_total_counts",
                                   "high_intention_counts",
                                   "low_intention_counts",
                                   "lost_connection_counts",
                                   "is_have_experience_lesson",
                                   "commu_counts",
                                   "high_intenetion_rate",
                                   "low_intenetion_rate",
                                   "lost_intenetion_rate",
                                   ]
                 )
    # gcforest
    gcforest(x_train, x_test, y_train, y_test,
             df_btest.drop("is_pigeon", axis=1), df_btest["is_pigeon"])

    # gcforest2
    gcforest2(x_train, x_test, y_train, y_test,
             df_btest.drop("is_pigeon", axis=1), df_btest["is_pigeon"])