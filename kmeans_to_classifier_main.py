from  data_treatment import load_data_yf,data_clean,seperate_label,data_seperate,load_data_new,data_transform_new,plot_eda,data_clean,feature_extend
from models import rf_mdoel,gbdt_mdoel,xgb_model,cat_boost_model,lgb_model,get_stacking
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from model_evalu import evalution_model,plot_importance
import numpy as npr
from xgboost import XGBClassifier
from sklearn.metrics import precision_score,f1_score,recall_score
import warnings
import pandas as pd
import numpy as np
import pymysql
from  catboost  import  CatBoostClassifier,CatBoostRegressor,Pool
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    #加载数据
    sql = "select * from bidata.trail_boost"
    df = load_data_new(sql, filename="df_201810166.csv")

    #加载新数据
    # sql_new = "select DISTINCT teacher_id,history_trail_cnt,history_trail_suc_cnt_bycontract,history_trail_suc_cnt_bystudent,first_tkod_tifl_count from trail_boost"
    # conn = pymysql.connect(host="rm-2ze974348wa9e1ev3uo.mysql.rds.aliyuncs.com", port=3306, user="yanfei_read",
    #                        passwd="xMbquuHi98JyfiF1", db="bidata", charset="utf8")
    # df_new = pd.read_sql(sql_new, conn)
    # conn.close()
    #
    # df = pd.merge(df,df_new,how="left",on=["teacher_id"])

    label_by_contract = "is_sucess_by_contract"
    label_by_pay = "is_sucess_by_pay"
    label_by_official_course = "is_sucess_by_official_course"
    labels = label_by_contract
    select_columns = [
        # 'student_no',
        'is_first_trail',
        # 'grade_rank',
        # 'teacher_id',
        # 'student_province',
        'student_province_byphone',
        'class_rank_fillna',
        'grade_subject',
        'student_grade',
        'student_city_class_detail',
        'know_origin_discretize',
        # 'coil_in_discretize',
        # #
        'subject_ids',
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
    #数据预处理
    df_train, df_btest= data_clean(df, min_date="2018-01-01", mid_date="2018-06-15", max_date="2018-06-30",label=labels)
    df_train = df_train[select_columns]
    df_btest = df_btest[select_columns]
    print(len(df_btest))
    print(df_train.columns)
    print('正/负', str(len(df_train[df_train[labels] == 1])) + '/' + str(len(df_train[df_train[labels] == 0])))
    t = len(df_train[df_train[labels] == 0]) / len(df_train[df_train[labels] == 1])
    v = len(df_btest[df_btest[labels] == 0]) / len(df_btest[df_btest[labels] == 1])
    print(t,v)

    # 特征筛选
    # from sklearn.feature_selection import RFECV
    #
    # dt_score = make_scorer(precision_score, pos_label=1)
    # rf = RandomForestClassifier(n_estimators=24, criterion='gini', max_depth=7,
    #                             random_state=5, class_weight={1: t},
    #                             n_jobs=-1)
    # selector = RFECV(rf, step=1, cv=5, scoring=dt_score, n_jobs=-1)
    # selector = selector.fit(df_train.drop([labels], axis=1), df_train[labels])
    #
    # print("查看哪些特征是被选择的", selector.support_)  # 查看哪些特征是被选择的
    # print("被筛选的特征数量", selector.n_features_)
    # print("特征排名", selector.ranking_)
    # columns = pd.DataFrame(df_train.drop([labels], axis=1).columns).rename(columns={0: "features"})
    # sl = pd.DataFrame(selector.support_).rename(columns={0: "result_rfecv"})
    # sk = pd.concat([columns, sl], axis=1)
    # sk_select = sk[sk['result_rfecv'] == True]
    # sm = list(sk_select["features"])
    # sm.append(labels)
    #
    # df_train = df_train[sm]
    # df_btest = df_btest[sm]
    # print(len(df_btest))


    #划分训练测试集
    X_train_tra, X_test_tra, df_btest= data_seperate(df_train,df_btest, size=0.3, cri=None,undeal_column=[
       # 'is_first_trail',
        # 'grade_rank',
        # 'teacher_id',
        # 'student_province',
        'student_province_byphone',
        # 'class_rank_fillna',
        'grade_subject',
        'student_grade',
        'student_city_class_detail',
        'know_origin_discretize',
        # 'coil_in_discretize',
        # #
        # 'subject_ids',
        # 'school_background',
        # 'student_sex_fillna',
        # 'teacher_sex',
      'coil_in',
      'know_origin',
        # "is_login",
        # "lesson_asigned_way",
        labels])

    # 划分label
    # x_train, y_train = seperate_label(X_train_tra, label=labels)
    # x_test, y_test = seperate_label(X_test_tra, label=labels)
    x_train = X_train_tra.copy()
    x_test = X_test_tra.copy()

    #sample_weigth
    y_train = x_train[labels]
    from collections import Counter
    cout = Counter(y_train)
    tt = cout[0] / cout[1]
    sample_weigh = np.where(y_train == 0, 1, tt)


    #k_means划分类别
    from sklearn.cluster import KMeans

    estimator = KMeans(n_clusters=5, random_state=0)  # 构造聚类器
    estimator.fit(x_train.drop(labels, axis=1))  # 聚类

    train_label = estimator.predict(x_train.drop(labels, axis=1))
    test_label = estimator.predict(x_test.drop(labels, axis=1))
    btest_label = estimator.predict(df_btest.drop("is_sucess_by_contract", axis=1))

    x_train["chunk_label"] = train_label
    x_test["chunk_label"] = test_label
    df_btest["chunk_label"] = btest_label
    # df_btest["count"] = 1

    # ss = pd.pivot_table(df_btest, index=["is_sucess_by_contract"], columns=["chunk_label"], values=["count"], aggfunc=np.sum)

    #rf0
    # clf = RandomForestClassifier(n_estimators=21, max_depth=5, max_features=9, random_state=5, n_jobs=-1,criterion="gini")
    # clf = GradientBoostingClassifier(loss="deviance", learning_rate=0.1,
    #                                   n_estimators=20, subsample=1.0,max_features=8,
    #                                   criterion="mse",warm_start=True,
    #                                   min_samples_split=2, min_samples_leaf=1,
    #                                  max_depth=5, random_state=5)
    # clf = XGBClassifier(
    #           max_depth=6,
    #           min_child_weight=1,
    #           learning_rate=0.1,
    #           n_estimators=20,
    #           silent=True,
    #           objective='binary:logistic',
    #           gamma=0,
    #           max_delta_step=0,
    #           subsample=1,
    #           colsample_bytree=1,
    #           colsample_bylevel=1,
    #           reg_alpha=0,
    #           reg_lambda=0,
    #           # scale_pos_weight=3.687,
    #           seed=1,
    #           missing=None,
    #           random_state=5)

    # clf= CatBoostClassifier(learning_rate=0.01, depth=9, l2_leaf_reg=0.1, loss_function='CrossEntropy',
    #                           # class_weights=[1, 2.8],
    #                           thread_count=24, random_state=5)

    from tpot import TPOTClassifier

    tpot_config = {
        'sklearn.ensemble.RandomForestClassifier':
            {
                'criterion': ['gini'],
                'n_estimators': range(20, 25),
                'max_depth': range(5, 10),
                'max_features': range(5, 10),
                'class_weight': [{1: i} for i in np.linspace(tt - 1, tt + 1, 3)]
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
            "scale_pos_weight": [i for i in np.linspace(tt - 1, tt + 1, 3)],
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
                                  np.linspace(tt - 1, tt + 1, 3)]},
        'lightgbm.LGBMModel': {
            'categorical_feature': ['auto'],
            # 'weight': sample_weigh,
            'boosting_type': ['gbdt', 'dart', 'rf'],
            'n_estimators': range(20, 25),
            'learning_rate ': [0.1, 0.01],
            'subsample_freq': [0.5, 0.8, 1],
            'colsample_bytree': [0.5, 0.8, 1],
            'num_leaves': range(28, 33),
        }
    }




    for i in range(5):
        tpo = TPOTClassifier(generations=10, verbosity=2, population_size=150,
                             scoring='f1', n_jobs=-1, config_dict=tpot_config,
                             mutation_rate=0.8, crossover_rate=0.2)
        x_train_x = np.array(x_train[x_train["chunk_label"] == i].drop(["chunk_label", labels],
                                                                 axis=1))
        x_test_x = np.array(x_test[x_test["chunk_label"] == i].drop(["chunk_label", labels],
                                                                 axis=1))
        df_btest_x = df_btest[df_btest["chunk_label"] == i].drop("chunk_label",
                                                                   axis=1)
        y_train_x = np.array(x_train[labels])
        # clf = tpo.fit(x_train_x, y_train_x)
        #
        # print(len(df_btest_x))
        # print("=========modelu", i, "============")
        # evalution_model(clf, df_btest_x.drop("is_sucess_by_contract", axis=1),
        #                 df_btest_x["is_sucess_by_contract"])
        #
        #
        # evalution_model(clf, df_btest.drop("is_sucess_by_contract", axis=1), df_btest["is_sucess_by_contract"])
        #
        #
        # #






















