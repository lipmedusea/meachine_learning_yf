from  data_treatment import load_data_yf,data_transform,seperate_label,data_seperate
from model_evalu import evalution_model,plot_importance
import numpy as np
from models import rf_mdoel,gbdt_mdoel,xgb_model,cat_boost_model
import pandas as pd
import multiprocessing
from model_evalu import SBS
pd.options.mode.chained_assignment = None
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pymysql
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from  data_treatment import load_data_yf,data_transform,seperate_label,data_seperate
from model_evalu import evalution_model,plot_importance
import numpy as np
from models import rf_mdoel,gbdt_mdoel,xgb_model,cat_boost_model
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score,f1_score,recall_score
from sklearn.base import clone
if __name__ == '__main__':
    # 加载数据
    sql = "select * from trial_course_success_rate_info"
    df = load_data_yf(sql)
    print(len(df))
    # 数据预处理
    df_train, df_btest = data_transform(df, min_date=None, mid_date="2018-04-11", max_date="2018-04-28")
    print('正/负', str(len(df_train[df_train['new_new_isSuccess'] == 1])) + '/' + str(
        len(df_train[df_train['new_new_isSuccess'] == 0])))
    t = len(df_train[df_train['new_new_isSuccess'] == 0]) / len(df_train[df_train['new_new_isSuccess'] == 1])
    # 划分label
    x, y = seperate_label(df_train, label="new_new_isSuccess")
    # 划分训练测试集
    x_train, x_test, y_train, y_test = data_seperate(x, y, size=0.3, cri=None, undeal_column=["studentNo", "teacherId"])
    dt_score = make_scorer(precision_score, pos_label=1)
    rf = RandomForestClassifier(n_estimators=24, criterion='gini', max_depth=7,
                                    random_state=0,class_weight={1:3.6},
                                    n_jobs=-1)
    #SBS
    # sbs = SBS(rf, k_features=9,test_size=0.3, random_state=0,scoring=f1_score)
    # sbs.fit(x.drop(["studentNo", "teacherId"], axis=1),y)
    # k_feat = [len(k) for k in sbs.subsets_]
    # import matplotlib.pyplot as plt
    # plt.plot(k_feat, sbs.scores_, marker='o')
    # plt.ylim([0.7, 1.1])
    # plt.ylabel('Accuracy')
    # plt.xlabel('Number of features')
    # plt.grid()
    # plt.tight_layout()
    # # plt.savefig('./sbs.png', dpi=300)
    # plt.show()




    #RFE
    # from sklearn.feature_selection import RFECV
    # dt_score = make_scorer(precision_score, pos_label=1)
    # rf = RandomForestClassifier(n_estimators=24, criterion='gini', max_depth=7,
    #                             random_state=0, class_weight={1: 3.6},
    #                             n_jobs=-1)
    # selector = RFECV(rf,step=1, cv=5,scoring=dt_score)
    # selector = selector.fit(x.drop(["studentNo", "teacherId"], axis=1),y)
    #
    # print("查看哪些特征是被选择的",selector.support_)#查看哪些特征是被选择的
    # print("被筛选的特征数量",selector.n_features_)
    # print("特征排名",selector.ranking_)
    # columns = pd.DataFrame(x.drop(["studentNo", "teacherId"],axis=1).columns).rename(columns={0:"features"})
    # sl=pd.DataFrame(selector.support_).rename(columns={0:"result_rfecv"})
    # sk = pd.concat([columns,sl],axis=1)
    # sk_select = sk[sk['result_rfecv']==True]

    from sklearn.feature_selection import RFECV
    x = df_train.copy()
    clf1 = RandomForestClassifier()
    clf2 = GradientBoostingClassifier()
    clf3 = XGBClassifier()
    dt_score = make_scorer(precision_score, pos_label=1)
    label = "new_new_isSuccess"
    unuse_column = ["studentNo", "teacherId"]
    models = [clf1],
    scoring = dt_score
    from collections import Counter
    y = x[label]
    cout = Counter(y)
    t = cout[0] / cout[1]
    sample_weigh = np.where(y_train == 0, 1, t)
    unuse_columns = [label] + unuse_column
    columns = pd.DataFrame(x.drop(unuse_columns, axis=1).columns).rename(columns={0: "features"})
    i = 1
    for clf in models:
        selector = RFECV(estimator=clf[0], step=1, cv=5, scoring=scoring,n_jobs=-1)
        selector = selector.fit(x.drop(unuse_columns, axis=1), y)
        neme = "model_" + str(i)
        sl = pd.DataFrame(selector.support_).rename(columns={0: neme})
        columns = pd.concat([columns, sl], axis=1)
        i = i + 1

    for colm in columns.columns[1:]:
        columns[colm] = np.where(columns[colm] == False, 0, 1)
    # columns = np.where(columns==True,1,0)

    sum = 0
    for j in range(len(models)):
        sum = sum + columns.iloc[:, j + 1]
    columns["sum"] = sum

    columns_select = columns[columns["sum"] > len(models) / 2]
    select_features = list(columns_select["features"])

    x_select = pd.concat(
        [pd.DataFrame(x[unuse_columns]).reset_index(drop=True), x[select_features].reset_index(drop=True)],
        axis=1)

    df_btest = pd.concat([df_btest[unuse_columns].reset_index(drop=True),
                          df_btest[select_features].reset_index(drop=True)], axis=1)