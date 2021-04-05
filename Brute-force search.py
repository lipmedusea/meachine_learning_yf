from  data_treatment import load_data_yf,data_transform,seperate_label,data_seperate,load_data_new,data_transform_new,plot_eda,data_clean,feature_extend
from model_evalu import evalution_model,plot_importance
import numpy as np
from models import rf_mdoel,gbdt_mdoel,xgb_model,cat_boost_model,lgb_model
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score,f1_score,recall_score
import warnings
import pandas as pd
from model_evalu import combine
from model_evalu import mul_model
import multiprocessing
from model_evalu import SBS
from multiprocessing import cpu_count
from  catboost  import  CatBoostClassifier,CatBoostRegressor,Pool
pd.options.mode.chained_assignment = None

import pymysql
import numpy as np
from sklearn.ensemble import RandomForestClassifier
if __name__ == '__main__':
    # df = load_data(sql='select * from returnSample')
    # 加载数据
    sql = "select * from bidata.trail_boost"
    df = load_data_new(sql, filename="df_201810166.csv")

    label_by_contract = "is_sucess_by_contract"
    label_by_pay = "is_sucess_by_pay"
    label_by_official_course = "is_sucess_by_official_course"
    labels = label_by_contract
    select_columns = [
        'student_no',
        'is_first_trail',
        # 'grade_rank',
        'teacher_id',
        # 'student_province',
        'student_province_byphone',
        'class_rank_fillna',
        'grade_subject',
        'student_grade',
        'student_city_class_detail',
        'know_origin_discretize',
        # 'coil_in_discretize',
        # #
        # 'subject_ids',
        'school_background',
        'student_sex_fillna',
        'teacher_sex',
        # "lesson_asigned_way",
        'coil_in',
        'know_origin',
        "is_login",

        #
        'history_rate_bystudent',
        'history_trail_suc_cnt_bycontract',
        'history_trail_suc_cnt_bystudent',
        'l3m_sucess_rate',
        'trial_course_rate',
        'l3m_student_relative',
        'score_mean',
        'daily_trail_count',
        'teacher_requirements_times',
        'sucess_rate',
        'l3m_trail_not_best_rate',
        'l3m_teacher_trail_not_best_cnt',
        'month',
        'l3m_trial_course_rate',
        'l3m_hw_submit_rate',
        'taught_trial_course_count',
        'first_tkod_tifl_count',
        'history_trail_cnt',
        'teacher_after_4d_lp_cnt',
        'l3m_hw_correct_rate',
        # #
        'teacher_fresh_hour',
        "effectiveCommunicationCount",
        "score_min",
        'learning_target_lenght',
        "teacher_staff_age_byopt",
        'self_evaluation_length',

        'l3m_avg_has_qz_lc',
        'l3m_avg_prop_has_qz_lc',
        'l3m_has_qz_lc',
        'l3m_prop_has_qz_lc',
        labels
    ]

    print(len(df))
    # 数据预处理
    df_train, df_btest = data_clean(df, min_date="2018-01-01", mid_date="2018-06-15", max_date="2018-06-30",
                                    label=labels)
    df_train = df_train[select_columns]
    df_btest = df_btest[select_columns]
    print(df_train.columns)
    print('正/负', str(len(df_train[df_train[labels] == 1])) + '/' + str(len(df_train[df_train[labels] == 0])))
    t = len(df_train[df_train[labels] == 0]) / len(df_train[df_train[labels] == 1])
    v = len(df_btest[df_btest[labels] == 0]) / len(df_btest[df_btest[labels] == 1])
    print(t, v)

    # 划分训练测试集
    X_train_tra, X_test_tra, df_btest = data_seperate(df_train, df_btest, size=0.3, cri="sigmod",
                                                      undeal_column=[
        'is_first_trail',
        # 'grade_rank',
        'teacher_id',
        # 'student_province',
        'student_province_byphone',
        'class_rank_fillna',
        'grade_subject',
        'student_grade',
        'student_city_class_detail',
        'know_origin_discretize',
        # 'coil_in_discretize',
        # #
        # 'subject_ids',
        'school_background',
        'student_sex_fillna',
        'teacher_sex',
      'coil_in',
      'know_origin',
        "is_login",
        # "lesson_asigned_way",
        labels])

    # 划分label
    # x, y = seperate_label(x_select, label=labels)
    x_train, y_train = seperate_label(X_train_tra, label=labels)
    x_test, y_test = seperate_label(X_test_tra, label=labels)
    #多进程

    # clf = RandomForestClassifier()
    # param_grids = {
    #                 'criterion': ['gini', 'entropy'],
    #                 'n_estimators': range(20, 23),
    #                 'max_depth': range(6, 11),
    #                 'max_features': range(6, 12),
    #                 'class_weight': [{1: i} for i in np.linspace(3, 5, 5)]
    #              }
    #
    # clf = GradientBoostingClassifier(loss="deviance", learning_rate=0.01,
    #                                   n_estimators=20, subsample=1.0,
    #                                   criterion="friedman_mse",
    #                                   min_samples_split=2, min_samples_leaf=1,
    #                                   max_depth=5, random_state=5)
    # param_grids = {"loss": ["deviance"],                   # GBDT parameters
    #               # "learning_rate": [0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0],
    #               "learning_rate": [0.02, 0.01],
    #               "n_estimators": range(20, 22),
    #               "subsample": [0.5, 0.8, 1.0],
    #               "criterion": ["friedman_mse", "mse"],
    #               "max_features": range(7, 10),       # DT parameters
    #               "max_depth": range(5, 8),
    #               # "min_samples_split": range(2, 10),
    #               # "min_samples_leaf": range(1, 10),
    #               # "min_weight_fraction_leaf": [0.08],    # Un-Tested parameters
    #               "warm_start": [True,False],
    #               # "min_impurity_decrease": [None],
    #               # "max_leaf_nodes": [None],
    #               }

    # clf = XGBClassifier(
    #     max_depth=7,
    #     min_child_weight=1,
    #     learning_rate=0.01,
    #     n_estimators=20,
    #     silent=True,
    #     objective='binary:logistic',
    #     gamma=0,
    #     max_delta_step=0,
    #     subsample=1,
    #     colsample_bytree=1,
    #     colsample_bylevel=1,
    #     reg_alpha=0,
    #     reg_lambda=0,
    #     seed=1,
    #     missing=None,
    #     random_state=5)
    #
    # param_grids = param_grid = {
    #     "learning_rate": [0.01,0.1],
    #     "n_estimators": range(20, 22),
    #     # "subsample": [0.5, 0.8, 1.0],  # 取多少样本，放过拟合
    #     "scale_pos_weight": [3.5,4,4.5,5],  # 类似class_weight
    #     # "max_features": range(7, 8),
    #     "max_depth": range(3, 8),
    #              }

    clf = CatBoostClassifier(
        learning_rate=0.01, depth=9, l2_leaf_reg=0.1,
        loss_function='Logloss', thread_count=18,
    )


    param_grids ={
        "learning_rate": [0.01,0.1,0.5],
        "n_estimators": range(20, 25),
        "loss_function": ['Logloss'],  # 取多少样本，放过拟合
        #     # "iterations": [40, 21],  # 类似class_weight
        'class_weights': [[1, i] for i in np.linspace(3, 6, 6)],
        "depth": range(5, 10),
        #     # "class_weights" :[1, 7],
    }


    import datetime
    starts_time = datetime.datetime.now()
    param_df = combine(param_grid=param_grids)
    numList = []
    score = 'recall'
    pool = multiprocessing.Pool(processes=32)
    for i in range(len(param_df)):
        param_df_one = param_df.iloc[i, :]
        numList.append(pool.apply_async(mul_model, (clf, x_train.drop(["student_no"], axis=1), y_train, df_btest.drop(["student_no", "is_sucess_by_contract"], axis=1),
                    df_btest["is_sucess_by_contract"], param_df_one, score)))
    result_list = [xx.get() for xx in numList]
    pool.close()
    pool.join()
    endstime = datetime.datetime.now()
    print(endstime - starts_time)
    df = result_list[0]
    for ds in result_list[1:]:
        df = pd.concat([df,ds],axis=0)
    df.to_csv('model_saved/adjust_paras_catboost_1016.csv', index=None, encoding='utf-8')

    print(df["f1_one"].max())


