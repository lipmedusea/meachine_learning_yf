import warnings
warnings.filterwarnings('ignore')
from  data_treatment import load_data_yf,data_clean,seperate_label,data_seperate,load_data_new,plot_eda,data_clean,feature_extend,data_clean2,feature_onehot
from models import rf_mdoel,gbdt_mdoel,xgb_model,cat_boost_model,lgb_model,get_stacking, MLPGradientCheck_model,lgb_sk_mdoel,gauss_navie_bayes
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
from models import  MLPGradientCheck

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    from sklearn.metrics import precision_score, f1_score, recall_score

    warnings.filterwarnings('ignore')
    # 加载数据
    sql = "SELECT * from bidata.trail_pigeon_wdf"
    df = load_data_new(sql, filename="df_20190223.csv")

    student_sql = """SELECT student_id,count(student_id) from trail_pigeon 
                            GROUP BY student_id"""
    student_mul = load_data_new(student_sql, filename="student_mul.csv")
    student_ids = student_mul[student_mul["count(student_id)"] == 1]

    df = df[df["student_id"].isin(student_mul["student_id"])]

    #載入cnn特徵
    cnn_df = pd.read_csv("load_data/cnn_features.csv")
    df = pd.merge(df, cnn_df, on="order_id", how="left")


    label_by_contract = "is_pigeon"
    labels = label_by_contract

    # # 试听课表
    # trail_df = load_data_new(sql, filename="trail_boost.csv")
    # # 合同状态3549
    # trail_df["contract_status_gpconcat"] = trail_df[
    #     "contract_status_gpconcat"].fillna("0")
    # trail_df["sx"] = [("3" in x) or ("4" in x) or ("5" in x) or ("9" in x) for x
    #                   in
    #                   trail_df["contract_status_gpconcat"]]
    # trail_df["is_sucess_by_contract"] = np.where(trail_df["sx"] == True, 1, 0)
    # trail_df = trail_df[["student_id", "is_sucess_by_contract"]]
    #
    # # 插入合同
    # df = pd.merge(df, trail_df, on="student_id", how="left")
    # df["is_sucess_by_contract"].fillna(0, inplace=True)

    select_columns = [
        "sale_id",
        "teacher_id",
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
        "cnn_prob"
    ]

    # 数据预处理
    df_train, df_btest = data_clean(df, min_date="2019-01-01", mid_date="2019-01-10", max_date="2019-01-15", label=labels)
    df_train = df_train[select_columns]
    df_btest = df_btest[select_columns]


    # 抽样
    # df_train = df_train.sample(n=None, frac=0.1, replace=False, weights=None,
    #                            random_state=0, axis=0)
    # df_btest = df_btest.sample(n=None, frac=0.5, replace=False, weights=None,
    #                            random_state=0, axis=0)


    print('正/负', str(len(df_train[df_train[labels] == 1])) + '/' + str(len(df_train[df_train[labels] == 0])))
    t = len(df_train[df_train[labels] == 0]) / len(df_train[df_train[labels] == 1])
    v = len(df_btest[df_btest[labels] == 0]) / len(df_btest[df_btest[labels] == 1])
    print(t, v)


    # #划分训练测试集
    X_train_tra, X_test_tra, df_btest= data_seperate(df_train, df_btest, size=0.3, label="is_pigeon", cri=None, undeal_column=None
      #   [
      #  # 'is_first_trail',
      #   # 'grade_rank',
      #   # 'teacher_id',
      #   # 'student_province',
      #   'student_province_byphone',
      #   # 'class_rank_fillna',
      #   # 'grade_subject',
      #   'student_grade',
      #   'student_city_class_detail',
      #   'know_origin_discretize',
      #   # 'coil_in_discretize',
      #   # #
      #   # 'subject_ids',
      #   # 'school_background',
      #   # 'student_sex_fillna',
      #   # 'teacher_sex',
      # 'coil_in',
      # 'know_origin',
      #   # "is_login",
      #   # "lesson_asigned_way",
      #   labels]
                                                     )

    X_train_tra.to_csv("load_data/x_train_yf.csv")


    # # 划分label
    print(df_train.columns)


    #onehot
    # X_train_tra = feature_onehot(X_train_tra, label=labels, features=["student_grade",
    #                     "know_origin_discretize",
    #                     "coil_in_discretize",
    #                     "know_origin",
    #                     "apply_user_id",
    #                     "student_province_byphone",
    #                     "with_certificate",
    #                     "subject_type",
    #                     "is_teacher_college",], condition=1)
    #
    # X_test_tra = feature_onehot(X_test_tra, label=labels,
    #                              features=["student_grade",
    #                                        "know_origin_discretize",
    #                                        "coil_in_discretize",
    #                                        "know_origin",
    #                                        "apply_user_id",
    #                                        "student_province_byphone",
    #                                        "with_certificate",
    #                                        "subject_type",
    #                                        "is_teacher_college", ], condition=1)

    #降采样
    # x0 = X_train_tra[X_train_tra[labels] == 0]
    # x1 = X_train_tra[X_train_tra[labels] == 1]
    # x0 = x0.sample(n=None, frac=0.33, replace=False, weights=None, random_state=0, axis=0)
    # X_train_tra = pd.concat([x0, x1], axis=0)


    #划分label
    x_train, y_train = seperate_label(X_train_tra, label=labels)
    x_test, y_test = seperate_label(X_test_tra, label=labels)
    print("x_train", x_train.shape)

    cout = Counter(y_train)
    tt = cout[0] / cout[1]
    sample_weigh = np.where(y_train == 0, 1, tt)

    x, y = make_classification(n_samples=1000, n_features=10, n_classes=2)

    #rf基模型
    def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
        clf = RandomForestClassifier(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=int(max_features),
                               # float
                               max_depth=int(max_depth),
                               random_state=5,
                               class_weight={1: 1.45},
                               criterion="gini"
                               )
        clf.fit(x_train, y_train)

        y_train_pred = clf.predict(x_train)
        y_test_pred = clf.predict(x_test)

        r1_x_train = round(f1_score(y_train, y_train_pred, average='micro'), 5) * 100
        r1_x_test = round(f1_score(y_test, y_test_pred, average='micro'), 5) * 100

        t = -r1_x_train + r1_x_test


        y_btest_pred = clf.predict(df_btest.drop("is_pigeon",axis=1))
        b_p1 = round(precision_score(df_btest["is_pigeon"], y_btest_pred, pos_label=1), 5) * 100
        b_r1 = round(recall_score(df_btest["is_pigeon"], y_btest_pred, pos_label=1), 5) * 100
        b_f1 = round(f1_score(df_btest["is_pigeon"], y_btest_pred, average='micro'), 5) * 100

        return b_f1

    rf_bo = BayesianOptimization(
        rf_cv,
        {
            'n_estimators': (10, 20),
            'min_samples_split': (2, 5),
            'max_features': (5, 10),
            'max_depth': (5, 10)
        },
        random_state = 6
    )

    rf_bo.maximize(
        init_points=10,
        n_iter=100,
    )

    fg = rf_bo.max

    clfs = RandomForestClassifier(n_estimators=int(fg["params"]["n_estimators"]),
                                 min_samples_split=int(fg["params"]["min_samples_split"]),
                                 max_features=int(fg["params"]["max_features"]),
                                 # float
                                 max_depth=int(fg["params"]["max_depth"]),
                                 random_state=5,
                                 class_weight={1: 1.46}
                                 )
    clfs.fit(x_train, y_train)

    print("================训练集================")
    evalution_model(clfs, x_train, y_train)
    print("================测试集================")
    evalution_model(clfs, x_test, y_test)
    print("===========b_test===================")
    evalution_model(clfs, df_btest.drop("is_pigeon",axis=1),df_btest["is_pigeon"])



    #catboost基模型
    # from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    #
    #
    # def catboost_cv(n_estimators, learning_rate, depth, l2_leaf_reg):
    #     clf = CatBoostClassifier(
    #                             n_estimators=int(n_estimators),
    #                             learning_rate=learning_rate,
    #                             depth=int(depth),
    #                             l2_leaf_reg=l2_leaf_reg,
    #                             loss_function='Logloss',
    #                             )
    #     clf.fit(x_train, y_train)
    #     y_train_pred = clf.predict(x_train)
    #     y_test_pred = clf.predict(x_test)
    #
    #     r1_x_train = round(f1_score(y_train, y_train_pred, pos_label=1), 5) * 100
    #     r1_x_test = round(f1_score(y_test, y_test_pred, pos_label=1),5) * 100
    #
    #     t = -r1_x_train + r1_x_test
    #
    #     y_btest_pred = clf.predict(df_btest.drop("is_pigeon",axis=1))
    #     f1_btest = round(f1_score(df_btest["is_pigeon"], y_btest_pred, pos_label=1),5) * 100
    #
    #     return f1_btest
    #
    #
    # catboost_bo = BayesianOptimization(
    #     catboost_cv,
    #     {
    #         'n_estimators': (10, 20),
    #         'learning_rate': (0.01, 0.2),
    #         'l2_leaf_reg': (0.1, 10),
    #         'depth': (5, 10)
    #     },
    #     random_state=6
    # )
    #
    # catboost_bo.maximize(
    #     init_points=10,
    #     n_iter=100,
    # )











