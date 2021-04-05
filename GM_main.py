import warnings
warnings.filterwarnings(action="ignore", category=ImportWarning, module="importlib*")
from data_treatment import load_data_yf,data_transform,seperate_label,data_seperate
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
    warnings.filterwarnings(action="ignore", category=ImportWarning, module="importlib*")
    from sklearn.metrics import precision_score, f1_score, recall_score
    warnings.filterwarnings('ignore')
    #加载数据
    sql = "select * from trial_course_success_rate_info"
    df = load_data_yf(sql)
    print(len(df))
    labels = "is_sucess_by_contract"
    # 数据预处理
    df_train, df_btest = data_transform(df, min_date="2017-07-01",
                                        mid_date="2018-04-11",
                                        max_date="2018-04-28")
    # 抽样
    # df_train = df_train.sample(n=None, frac=0.1, replace=False, weights=None,
    #                            random_state=0, axis=0)
    # df_btest = df_btest.sample(n=None, frac=0.5, replace=False, weights=None,
    #                            random_state=0, axis=0)
    print('正/负', str(
        len(df_train[df_train['is_sucess_by_contract'] == 1])) + '/' + str(
        len(df_train[df_train['is_sucess_by_contract'] == 0])))
    t = len(df_train[df_train['is_sucess_by_contract'] == 0]) / len(
        df_train[df_train['is_sucess_by_contract'] == 1])
    #
    select_columns = [
        # 'student_no',
        'teacher_id',
        'province_byphone',
        'new_studentSource',
        'firstTrialCourse',
        'seniorSchoolInfo',
        'new_studentGrade',
        'grade_subject',
        'new_classRank',
        'new_new_studentSex',
        'teacherSex',
        'new_student_City_Class',
        'new_teacherPlanCourseInterval',
        'teachedTrialCourseCount',
        'effectiveCommunicationCount',
        'score_mean',
        'score_min',
        'new_learningTarget',
        'teacher_audition_success_rate',
        'new_new_teached_age',
        'new_teacher_daily_audition_count',
        'new_teacher_daily_audition_success_count',
        'new_new_selfEvaluation',
        'new_new_rate',
        'new_coil_in',
        'lm_informal_teached_lesson_count',
        'taught_total_time',
        'first_tkod_tifl_count',
        'teachedTrialSuccessStudentCount',
        labels
    ]

    # 抽样
    # df_train = df_train.sample(n=None, frac=0.1, replace=False, weights=None,
    #                            random_state=0, axis=0)


    print(df_train.columns)
    print('正/负', str(len(df_train[df_train[labels] == 1])) + '/' + str(len(df_train[df_train[labels] == 0])))
    t = len(df_train[df_train[labels] == 0]) / len(df_train[df_train[labels] == 1])
    v = len(df_btest[df_btest[labels] == 0]) / len(df_btest[df_btest[labels] == 1])
    print(t,v)





    #划分训练测试集
    X_train_tra, X_test_tra, df_btest= data_seperate(df_train,df_btest, size=0.3, cri=None,undeal_column=None)


    # 划分label
    # x, y = seperate_label(x_select, label=labels)
    x_train, y_train = seperate_label(X_train_tra, label=labels)
    x_test, y_test = seperate_label(X_test_tra, label=labels)

    # 遗传算法
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    cout = Counter(y_train)
    tt = cout[0] / cout[1]
    sample_weigh = np.where(y_train == 0, 1, tt)
    tpot_config = {
        'sklearn.ensemble.RandomForestClassifier':
            {
                'criterion': ['gini'],
                'n_estimators': range(20, 25),
                'max_depth': range(5, 10),
                'max_features': range(5, 10),
                'class_weight': [{1: i} for i in np.linspace(tt-2, tt+4, 5)]
            },
        'sklearn.ensemble.GradientBoostingClassifier': {
            "loss": ["deviance"],  # GBDT parameters
            "learning_rate": [0.01, 0.1],
            "n_estimators": range(20, 25),
            "subsample": [0.5, 0.8, 1.0],
            "criterion": ["friedman_mse", "mse"],
            "max_features": range(5, 10),  # DT parameters
            "max_depth": range(5, 10),
            "warm_start": [True]},
        'xgboost.XGBClassifier': {
            "learning_rate": [0.1, 0.01],
            "n_estimators": range(20, 25),
            "scale_pos_weight": [i for i in np.linspace(tt-2, tt+4, 5)],  # 类似class_weight
            "subsample": [0.85],  # 取多少样本，放过拟合
            "min_child_weight": range(6, 7),
            "max_depth": range(3, 8),
        },
        'catboost.CatBoostClassifier':
            {
                "learning_rate": [0.01],
                "loss_function": ['CrossEntropy', 'Logloss'],
                "depth": range(9, 10),
                "class_weights": [[1, i] for i in np.linspace(tt-2, tt+4, 5)]},
        'lightgbm.LGBMModel': {
            'categorical_feature':['auto'],
            # 'weight': sample_weigh,
            'boosting_type': ['gbdt', 'dart', 'rf'],
            'n_estimators': range(20, 25),
            'learning_rate ': [0.1, 0.01],
            'subsample_freq': [0.5, 0.8, 1],
            'colsample_bytree': [0.5, 0.8, 1],
            'num_leaves': range(28, 33),
        }
    }

    from sklearn.metrics import make_scorer

    dt_score = make_scorer(f1_score, pos_label=1)

    tpo = TPOTClassifier(generations=50, verbosity=2, population_size=100,
                         scoring='precision', n_jobs=-1, config_dict=tpot_config)

    tpo.fit(x_train, y_train,
            # sample_weight=sample_weigh
            )
    evalution_model(tpo, x_train, y_train)
    evalution_model(tpo, x_test, y_test)
    evalution_model(tpo, np.array(df_btest.drop("is_sucess_by_contract", axis=1)),
                    df_btest["is_sucess_by_contract"])

    import datetime

    i = datetime.datetime.today()
    print(i)
    s = str(i.month) + str(i.day) + str(i.hour) + str(i.minute)
    model_name = "GM_export/main/" + "GM" + s + ".py"
    tpo.export(model_name)

    # exported_pipeline = XGBClassifier(CombineDFs(input_matrix, CombineDFs(input_matrix,
    #                                                   CombineDFs(input_matrix,
    #                                                              XGBClassifier(
    #                                                                  input_matrix,
    #                                                                  learning_rate=0.1,
    #                                                                  max_depth=4,
    #                                                                  min_child_weight=6,
    #                                                                  n_estimators=24,
    #                                                                  scale_pos_weight=3.6872771474878445,
    #                                                                  subsample=0.85)))),
    #               learning_rate=0.1, max_depth=7, min_child_weight=6,
    #               n_estimators=24, scale_pos_weight=3.6872771474878445,
    #               subsample=0.85)
    #
    # cout = Counter(y_train)
    # tt = cout[0] / cout[1]
    # sample_weigh = np.where(y_train == 0, 1, tt)
    # exported_pipeline.fit(x_train, y_train)
    # evalution_model(exported_pipeline, x_train, y_train)
    # evalution_model(exported_pipeline, x_test, y_test)
    # evalution_model(exported_pipeline,
    #                 df_btest.drop("is_sucess_by_contract", axis=1),
    #                 df_btest["is_sucess_by_contract"])