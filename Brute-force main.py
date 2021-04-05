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
          # 加载数据
          sql = "select * from trial_course_success_rate_info"
          df = load_data_yf(sql)
          print(len(df))
          labels = "is_sucess_by_contract"
          # 数据预处理
          df_train, df_btest = data_transform(df, min_date="2017-07-01", mid_date="2018-04-11", max_date="2018-04-28")
          print('正/负', str(len(df_train[df_train['is_sucess_by_contract'] == 1])) + '/' + str(
                    len(df_train[df_train['is_sucess_by_contract'] == 0])))
          t = len(df_train[df_train['is_sucess_by_contract'] == 0]) / len(
                    df_train[df_train['is_sucess_by_contract'] == 1])
          #
          select_columns = [
                    'student_no',
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

          # 特征筛选
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

          # 划分训练测试集
          df_train = df_train[select_columns]
          df_btest = df_btest[select_columns]
          print(df_train.columns)
          print('正/负', str(len(df_train[df_train[labels] == 1])) + '/' + str(len(df_train[df_train[labels] == 0])))
          t = len(df_train[df_train[labels] == 0]) / len(df_train[df_train[labels] == 1])
          v = len(df_btest[df_btest[labels] == 0]) / len(df_btest[df_btest[labels] == 1])
          print(t, v)

          X_train_tra, X_test_tra, df_btest = data_seperate(df_train, df_btest, size=0.3, cri=None,
                                                            undeal_column=["student_no", "teacher_id"])
          # 划分label
          x_train, y_train = seperate_label(X_train_tra, label=labels)
          x_test, y_test = seperate_label(X_test_tra, label=labels)

          pred_prob_train, pred_prob_test, pred_prob_btest = rf_mdoel(x_train, x_test, y_train, y_test, df_btest)
          x_train = pd.concat([x_train, pred_prob_train], axis=1).rename(columns={0: "pred_0", 1: "pred_1"})
          x_test = pd.concat([x_test, pred_prob_test], axis=1).rename(columns={0: "pred_0", 1: "pred_1"})
          df_btest = pd.concat([df_btest, pred_prob_btest], axis=1).rename(columns={0: "pred_0", 1: "pred_1"})

          # 多进程

          # clf = RandomForestClassifier()
          # param_grids = {
          #                 'criterion': ['gini', 'entropy'],
          #                 'n_estimators': range(20, 23),
          #                 'max_depth': range(6, 11),
          #                 'max_features': range(6, 12),
          #                 'class_weight': [{1: i} for i in np.linspace(2, 4, 6)]
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
          # param_grids  = {
          #     "learning_rate": [0.01,0.1],
          #     "n_estimators": range(20, 22),
          #     # "subsample": [0.5, 0.8, 1.0],  # 取多少样本，放过拟合
          #     "scale_pos_weight": [i for i in np.linspace(2, 4, 6)],  # 类似class_weight
          #     # "max_features": range(7, 8),
          #     "max_depth": range(3, 8)
          #              }

          clf = CatBoostClassifier(
                    learning_rate=0.01, depth=9, l2_leaf_reg=0.1,
                    loss_function='Logloss', thread_count=18,
          )

          param_grids = {
                    "learning_rate": [0.01, 0.1, 0.5],
                    "n_estimators": range(20, 25),
                    "loss_function": ['Logloss'],  # 取多少样本，放过拟合
                    #     # "iterations": [40, 21],  # 类似class_weight
                    'class_weights': [[1, i] for i in np.linspace(2, 4, 6)],
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
                    numList.append(pool.apply_async(mul_model, (clf, x_train.drop(["student_no"], axis=1), y_train,
                                                                df_btest.drop(["student_no", "is_sucess_by_contract"],
                                                                              axis=1),
                                                                df_btest["is_sucess_by_contract"], param_df_one,
                                                                score)))
          result_list = [xx.get() for xx in numList]
          pool.close()
          pool.join()
          endstime = datetime.datetime.now()
          print(endstime - starts_time)
          df = result_list[0]
          for ds in result_list[1:]:
                    df = pd.concat([df, ds], axis=0)
          df.to_csv('model_saved/adjust_main_cat_1017.csv', index=None, encoding='utf-8')

          print(df["f1_one"].max())



