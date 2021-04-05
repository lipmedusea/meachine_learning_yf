from data_treatment import load_data_yf,data_transform,seperate_label,data_seperate
from model_evalu import evalution_model,plot_importance
import numpy as np
from models import rf_mdoel,gbdt_mdoel,xgb_model,cat_boost_model,lgb_model,gauss_navie_bayes,MLPGradientCheck_model
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score,f1_score,recall_score
import pandas as pd
from models import get_stacking
from tpot import TPOTClassifier
from sklearn.linear_model import LogisticRegression
if __name__ == '__main__':
    #加载数据
    sql = "select * from trial_course_success_rate_info"
    df = load_data_yf(sql)
    print(len(df))
    labels="is_sucess_by_contract"
    #数据预处理
    df_train, df_btest = data_transform(df, min_date="2017-07-01", mid_date="2018-04-11", max_date="2018-04-28")
    #抽样
    df_train = df_train.sample(n=None, frac=0.5, replace=False, weights=None,
                     random_state=0, axis=0)
    # df_btest = df_btest.sample(n=None, frac=0.5, replace=False, weights=None,
    #                            random_state=0, axis=0)
    print('正/负', str(len(df_train[df_train['is_sucess_by_contract'] == 1])) + '/' + str(len(df_train[df_train['is_sucess_by_contract'] == 0])))
    t = len(df_train[df_train['is_sucess_by_contract'] == 0]) / len(df_train[df_train['is_sucess_by_contract'] == 1])
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



    #特征筛选
    from sklearn.feature_selection import RFECV

    # dt_score = make_scorer(precision_score, pos_label=1)
    # rf = RandomForestClassifier(n_estimators=24, criterion='gini', max_depth=7,
    #                             random_state=0, class_weight={1: 3.6},
    #                             n_jobs=-1)
    # selector = RFECV(rf, step=1, cv=5, scoring=dt_score,n_jobs=-1)
    # selector = selector.fit(x.drop(["student_no", "teacher_id"], axis=1), y)
    #
    # print("查看哪些特征是被选择的", selector.support_)  # 查看哪些特征是被选择的
    # print("被筛选的特征数量", selector.n_features_)
    # print("特征排名", selector.ranking_)
    # columns = pd.DataFrame(x.drop(["student_no", "teacher_id"], axis=1).columns).rename(columns={0: "features"})
    # sl = pd.DataFrame(selector.support_).rename(columns={0: "result_rfecv"})
    # sk = pd.concat([columns, sl], axis=1)
    # sk_select = sk[sk['result_rfecv'] == True]
    #
    # x_select = pd.concat([x[["student_no", "teacher_id"]].reset_index(drop=True),x[list(sk_select["features"])].reset_index(drop=True)],axis=1)

    # from models import majorvote_rfecv_by_models
    # clf1 =RandomForestClassifier(n_estimators=24, criterion='gini', max_depth=7,
    #                             random_state=0, class_weight={1: 2.7}
    #                             )
    # clf2 =GradientBoostingClassifier()
    # clf3 = XGBClassifier(scale_pos_weight=3.6)
    # dt_score = make_scorer(precision_score, pos_label=1)
    # x_select, df_btest,select_features = majorvote_rfecv_by_models(df_train,df_btest,label="is_sucess_by_contract",
    #                                                                   unuse_column=["student_no"],
    #                                                                 models=[clf1],scoring=dt_score,filename="select_features.csv")


    #划分训练测试集
    df_train = df_train[select_columns]
    df_btest = df_btest[select_columns]
    print(df_train.columns)
    print('正/负', str(len(df_train[df_train[labels] == 1])) + '/' + str(len(df_train[df_train[labels] == 0])))
    t = len(df_train[df_train[labels] == 0]) / len(df_train[df_train[labels] == 1])
    v = len(df_btest[df_btest[labels] == 0]) / len(df_btest[df_btest[labels] == 1])
    print(t, v)


    X_train_tra, X_test_tra, df_btest = data_seperate(df_train, df_btest, size=0.3, cri=None,undeal_column=["teacher_id"])
    # 划分label
    x_train, y_train = seperate_label(X_train_tra, label=labels)
    x_test, y_test = seperate_label(X_test_tra, label=labels)

    # x_train = np.array(x_train)


    # 遗传算法
    # from sklearn.metrics import make_scorer
    #
    # dt_score = make_scorer(f1_score, pos_label=1)
    # from collections import Counter
    #
    # tpo = TPOTClassifier(generations=20, verbosity=2, population_size=100,
    #                      scoring=dt_score, n_jobs=-1)
    # cout = Counter(y_train)
    # tt = cout[0] / cout[1]
    # sample_weigh = np.where(y_train == 0, 1, tt)
    # tpo.fit(x_train, y_train, sample_weight=sample_weigh)
    # evalution_model(tpo, x_train, y_train)
    # evalution_model(tpo, x_test, y_test)
    # evalution_model(tpo, df_btest.drop("is_sucess_by_contract", axis=1),
    #                 df_btest["is_sucess_by_contract"])
    #
    # import time
    #
    # model_name = "GM_export/main/" + "GM" + str(
    #     time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + ".py"
    # tpo.export(model_name)




    # clf1 = RandomForestClassifier(n_estimators=24, max_depth=5, max_features=5, random_state=5, n_jobs=-1)
    # clf2 = GradientBoostingClassifier(loss="deviance", learning_rate=0.1,max_features=7,
    #                                   n_estimators=21, subsample=1.0,
    #                                   criterion="friedman_mse",
    #                                   min_samples_split=2, min_samples_leaf=1,
    #                                   max_depth=7, random_state=5,warm_start=True)
    #
    # clf3 = XGBClassifier(
    #     max_depth=7,
    #     min_child_weight=1,
    #     learning_rate=0.1,
    #     n_estimators=21,
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
    #
    # train_sets = []
    # test_sets = []
    # btest_sets = []
    # for clf in [clf1,clf2,clf3]:
    #     train_set, test_set, btest_set = get_stacking(clf, x_train, y_train, x_test, df_btest, n_folds=5)
    #     train_sets.append(train_set)
    #     test_sets.append(test_set)
    #     btest_sets.append(btest_set)
    #
    # x_train = pd.DataFrame(np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)).rename(columns={0:"rf_1",1:"gbdt_1",2:"xgb_1"})
    # x_test = pd.DataFrame(np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)).rename(columns={0:"rf_1",1:"gbdt_1",2:"xgb_1"})
    # df_xbtest = pd.DataFrame(np.concatenate([y_btest_set.reshape(-1, 1) for y_btest_set in btest_sets], axis=1)).rename(columns={0:"rf_1",1:"gbdt_1",2:"xgb_1"})
    #









    #模型训练
    # rf模型
    # df_btest = pd.concat([df_btest[["student_no", "teacher_id","is_sucess_by_contract"]].reset_index(drop=True),df_btest[list(sk_select["features"])].reset_index(drop=True)],axis=1)
    rf_prob_train, rf_prob_test, rf_prob_btest=rf_mdoel(x_train,x_test,y_train,y_test,df_btest.drop("is_sucess_by_contract",axis=1),df_btest["is_sucess_by_contract"],rename=["rf_0","rf_1"])

    # x_train = pd.concat([x_train, pred_prob_train], axis=1)
    # x_test = pd.concat([x_test, pred_prob_test], axis=1)
    # df_btest = pd.concat([df_btest, pred_prob_btest], axis=1)
    #gbdt模型
    gbdt_prob_train, gbdt_prob_test, gbdt_prob_btest =gbdt_mdoel(x_train, x_test, y_train, y_test, df_btest.drop("is_sucess_by_contract",axis=1),df_btest["is_sucess_by_contract"])

    # gbdt模型
    xgb_prob_train, xgb_prob_test, xgb_prob_btest = xgb_model(x_train, x_test, y_train, y_test, df_btest.drop("is_sucess_by_contract",axis=1),df_btest["is_sucess_by_contract"])

    # x_train = pd.concat([x_train, rf_prob_train,gbdt_prob_train,xgb_prob_train], axis=1)[["student_no","rf_1","gbdt_1","xgb_1"]]
    # x_test = pd.concat([x_test, rf_prob_btest,gbdt_prob_test,xgb_prob_test], axis=1)[["student_no","rf_1","gbdt_1","xgb_1"]]
    # df_btest = pd.concat([df_btest, rf_prob_btest,gbdt_prob_btest,xgb_prob_btest], axis=1)[["student_no","rf_1","gbdt_1","xgb_1","is_sucess_by_contract"]]

    # x_train = pd.concat([x_train, rf_prob_train, gbdt_prob_train, xgb_prob_train], axis=1)
    # x_test = pd.concat([x_test, rf_prob_btest, gbdt_prob_test, xgb_prob_test], axis=1)
    # df_btest = pd.concat([df_btest, rf_prob_btest, gbdt_prob_btest, xgb_prob_btest], axis=1)

    # rf_prob_train, rf_prob_test, rf_prob_btest = rf_mdoel(x_train, x_test, y_train, y_test, df_btest,rename=["rf_00","rf_11"])
    # #
    # # #lgb模型
    lgb_model(x_train, x_test, y_train, y_test, df_btest.drop("is_sucess_by_contract",axis=1),df_btest["is_sucess_by_contract"])
    #
    # #catboost
    #
    cat_boost_model(x_train, x_test, y_train, y_test, df_btest.drop("is_sucess_by_contract",axis=1),df_btest["is_sucess_by_contract"])
    #
    #major_vote
    from models import major_vote_model
    # major_vote_model(x_train, x_test, y_train, y_test, df_btest, model_weight=[0.2, 0.2, 0.2, 0.4], boundary=0.5)


    #gauss_navie_bayes
    # gauss_navie_bayes(x_train, x_test, y_train, y_test, df_btest.drop("is_sucess_by_contract",axis=1),df_btest["is_sucess_by_contract"])

    #B
    # MLPGradientCheck_model(np.array(x_train), np.array(x_test), y_train, y_test,
    #                     np.array(df_btest.drop("is_sucess_by_contract", axis=1)),
    #                   df_btest["is_sucess_by_contract"])
