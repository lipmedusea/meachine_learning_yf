from  data_treatment import load_data_yf,data_clean,seperate_label,data_seperate,load_data_new,data_transform_new,plot_eda,data_clean,feature_extend
from models import rf_mdoel,gbdt_mdoel,xgb_model,cat_boost_model,lgb_model,get_stacking
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from model_evalu import evalution_model,plot_importance
import numpy as np
from xgboost import XGBClassifier
from catboost  import  CatBoostClassifier,CatBoostRegressor,Pool

import tpot
from tpot import TPOTClassifier
from sklearn.metrics import precision_score,f1_score,recall_score
import warnings
import pandas as pd
import numpy as np
import pymysql
from collections import Counter
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

if __name__ == '__main__':
    from sklearn.metrics import precision_score, f1_score, recall_score
    warnings.filterwarnings('ignore')
    #加载数据
    sql = "select * from bidata.trail_boost"
    df = load_data_new(sql, filename="df_20190108.csv")

    label_by_contract = "is_sucess_by_contract"
    label_by_pay = "is_sucess_by_pay"
    label_by_official_course = "is_sucess_by_official_course"
    labels = label_by_contract
    select_columns = [
        "his_success_rate",
        "history_rate_bystudent",
        "student_grade",
        "know_origin_discretize",
        "history_rate_bycontract",
        "coil_in_discretize",
        "l3m_sucess_rate",
        "before_20181201_sale_trial_suc_count",
        "know_origin",
        "sucess_rate",
        "score_mean",
        "sale_trial_total_count",
        "score_min",
        "lm_sale_success_rate",
        "history_trail_suc_cnt_bycontract",
        "apply_user_id",
        "student_city_class_detail",
        "sale_success_rate",
        "day",
        "student_province_byphone",
        'is_sucess_by_contract',
        # "isin_activity",
        # "sale_work_age",

        "teacher_work_year",
        "exam_scores",
        "with_certificate",
        "subject_type",
        "is_teacher_college",

        "l3m_taught_trial_student_cnt",
        "l3m_teacher_submit_hw_cnt",
        "l3m_teacher_hw_cnt",
        "l3m_teacher_correct_hw_cnt",
        "l3m_hw_submit_rate",
        "l3m_hw_correct_rate",
        "l3m_student_relative",
        "l3m_taught_official_student_cnt",

        # 'teacher_after_4d_lp_cnt',
        # "coop_times_saler_teacher",
        'student_city_class_detail',
        # 'student_city_class',
        'teacher_city_class_detail',

        #
        # "week_day",
        "class_hour",

        # "city_class_cross",
        # "sex_cross",
        # "province_cross",

        # "teacher_month_count",
        "month_off_rate",
        # "teacher_month_off_count",
        # "teacher_month_tri_count",
        # "teacher_job_type",

    ]

    print(len(df))
    #数据预处理
    df_train, df_btest= data_clean(df, min_date="2018-06-01", mid_date="2018-09-15", max_date="2018-09-30", label=labels)
    df_train = df_train[select_columns]
    df_btest = df_btest[select_columns]

    print('正/负', str(len(df_train[df_train[labels] == 1])) + '/' + str(
        len(df_train[df_train[labels] == 0])))
    t = len(df_train[df_train[labels] == 0]) / len(
        df_train[df_train[labels] == 1])
    v = len(df_btest[df_btest[labels] == 0]) / len(
        df_btest[df_btest[labels] == 1])

    # 抽样
    # df_train = df_train.sample(n=None, frac=0.1, replace=False, weights=None,
    #                            random_state=0, axis=0)
    # df_btest = df_btest.sample(n=None, frac=0.5, replace=False, weights=None,
    #                            random_state=0, axis=0)

    # print(df_train.columns)
    # print('正/负', str(len(df_train[df_train[labels] == 1])) + '/' + str(len(df_train[df_train[labels] == 0])))
    # t = len(df_train[df_train[labels] == 0]) / len(df_train[df_train[labels] == 1])
    # v = len(df_btest[df_btest[labels] == 0]) / len(df_btest[df_btest[labels] == 1])
    # print(t,v)


    # #划分训练测试集
    X_train_tra, X_test_tra, df_btest= data_seperate(df_train,df_btest, size=0.3, cri=None,undeal_column=None
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

    # # 划分label
    # # x, y = seperate_label(x_select, label=labels)
    x_train, y_train = seperate_label(X_train_tra, label=labels)
    x_test, y_test = seperate_label(X_test_tra, label=labels)

    # #遗传算法
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    cout = Counter(y_train)
    tt = cout[0] / cout[1] + 0.4
    sample_weigh = np.where(y_train == 0, 1, tt)
    tpot_config = {
        'sklearn.ensemble.RandomForestClassifier':
            {
                'criterion': ['gini'],
                'n_estimators': range(20, 25),
                'max_depth': range(5, 10),
                'max_features': range(5, 10),
                'class_weight': [{1: i} for i in np.linspace(4.16, 4.17, 1)]
            },
        # 'sklearn.ensemble.GradientBoostingClassifier': {
        #     "loss": ["deviance"],  # GBDT parameters
        #     "learning_rate": [0.01, 0.1],
        #     "n_estimators": range(20, 25),
        #     "subsample": [0.5, 0.8, 1.0],
        #     "criterion": ["friedman_mse", "mse"],
        #     "max_features": range(5, 10),  # DT parameters
        #     "max_depth": range(5, 10),
        #     "warm_start": [True]},

        'xgboost.XGBClassifier': {
            "learning_rate": [0.1, 0.01],
            "n_estimators": range(20, 25),
            "scale_pos_weight": [i for i in np.linspace(4.16, 4.17, 1)],
        # 类似class_weight
            "subsample": [0.85],  # 取多少样本，放过拟合
            "min_child_weight": range(6, 7),
            "max_depth": range(3, 8),
        },

        'catboost.CatBoostClassifier':
            {
                "learning_rate": [0.01],
                "loss_function": ['CrossEntropy', 'Logloss'],  # 取多少样本，放过拟合
                "depth": range(9, 10),
                "class_weights": [[1, i] for i in
                                  np.linspace(4.16, 4.17, 1)]
                },
        # 'lightgbm.LGBMModel': {
        #     'categorical_feature': ['auto'],
        #     # 'weight': sample_weigh,
        #     'boosting_type': ['gbdt', 'dart', 'rf'],
        #     'n_estimators': range(20, 25),
        #     'learning_rate ': [0.1, 0.01],
        #     'subsample_freq': [0.5, 0.8, 1],
        #     'colsample_bytree': [0.5, 0.8, 1],
        #     'num_leaves': range(28, 33),
        #                        },

        'lightgbm.LGBMClassifier': {
            "learning_rate": [0.1, 0.01],
                "n_estimators": range(20, 25),
                # "max_depth ": range(5, 6),
                 # "boosting_type": ["gbdt", "rf"],
                 "class_weight": [{1: 4.16}],
                 "subsample": [1, 0.85]
                              }
    }
    from sklearn.metrics import precision_score, f1_score, recall_score
    from sklearn.metrics.scorer import make_scorer
    dt_score = make_scorer(f1_score, pos_=1)


    def my_pred(y_true, y_pred):
        t1 = ((y_pred == y_true) & (y_pred == 1)).sum()
        t0 = ((y_pred == y_true) & (y_pred == 0)).sum()
        f1 = ((y_pred != y_true) & (y_pred == 1)).sum()
        f0 = ((y_pred != y_true) & (y_pred == 0)).sum()

        recall = t1 / (t1 + f1)
        pre = t1 / (t1 + f0)
        f1 = 2 * pre * recall / (recall + pre)
        return f1


    my_custom_scorer = make_scorer(my_pred, greater_is_better=True)

    tpo = TPOTClassifier(generations=100, verbosity=2, population_size=100,
                         scoring = 'f1', n_jobs=-1, config_dict=tpot_config,
                         mutation_rate=0.9, crossover_rate=0.1, cv=5, random_state = 5)

    tpo.fit(x_train, y_train)

    evalution_model(tpo, x_train, y_train)
    evalution_model(tpo, x_test, y_test)
    evalution_model(tpo, np.array(df_btest.drop("is_sucess_by_contract",axis=1)),df_btest["is_sucess_by_contract"])

    import datetime
    i = datetime.datetime.today()
    print(i)
    s = str(i.month) + str(i.day) + str(i.hour) + str(i.minute)
    model_name = "GM_export/main_new/"+"GM"+s+".py"
    tpo.export(model_name)



    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline, make_union
    from tpot.builtins import StackingEstimator
    from xgboost import XGBClassifier
    from sklearn.preprocessing import FunctionTransformer
    from copy import copy

    exported_pipeline = make_pipeline(
        make_union(
            FunctionTransformer(copy),
            FunctionTransformer(copy)
        ),
        StackingEstimator(
            estimator=XGBClassifier(learning_rate=0.1, max_depth=5,
                                    min_child_weight=6, n_estimators=21,
                                    scale_pos_weight=4.16, subsample=0.85)),
        RandomForestClassifier(class_weight={1: 4.16}, criterion="gini",
                               max_depth=8, max_features=6, n_estimators=23)
    )

    exported_pipeline.fit(x_train, y_train)
    evalution_model(exported_pipeline, x_train, y_train)
    evalution_model(exported_pipeline, x_test, y_test)
    evalution_model(exported_pipeline,
                    np.array(df_btest.drop("is_sucess_by_contract", axis=1)),
                    df_btest["is_sucess_by_contract"])


