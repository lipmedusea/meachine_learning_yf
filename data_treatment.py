import pandas as pd
import os
import pymysql
import math
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder


def load_data_yf(sql):
    save_dir = 'load_data/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    load_path = save_dir
    filenames = os.listdir(load_path)
    if "df_load.csv" not in filenames:
        # conn = pymysql.connect(host="bi-private.hfjy.com", port=33333, user="data_hfjy", passwd="data_hfjy_123456",
        #                        db="hfSB", charset="udf8")
        conn = pymysql.connect(host="rm-2ze974348wa9e1ev3uo.mysql.rds.aliyuncs.com", port=3306, user="yanfei_read",
                               passwd="xMbquuHi98JyfiF1", db="bidata", charset="udf8")
        df = pd.read_sql(sql, conn)
        conn.close()
        df.to_csv('load_data/df_load.csv', index=None, encoding="GB18030")
    else:
        df = pd.read_csv('load_data/df_load.csv', encoding="GB18030", low_memory=False)
    return df

def load_data_new(sql,filename="df_load_new.csv"):
    save_dir = 'load_data/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    load_path = save_dir
    filenames = os.listdir(load_path)
    filepath = save_dir + filename
    if filename not in filenames:
        # conn = pymysql.connect(host="bi-private.hfjy.com", port=33333, user="data_hfjy", passwd="data_hfjy_123456",
        #                        db="hfSB", charset="udf8")
        conn = pymysql.connect(host="rm-2ze974348wa9e1ev3uo.mysql.rds.aliyuncs.com", port=3306, user="yanfei_read",
                               passwd="DxIqy0R9rzT95yBfstgx", db="bidata", charset="utf8")
        df = pd.read_sql(sql, conn)
        conn.close()
        df.to_csv(filepath, index=None, encoding="GB18030")
    else:
        df = pd.read_csv(filepath, encoding="GB18030")
    return df


import numpy as np

def data_clean(df, min_date=None, mid_date="2018-04-11", max_date=None,label=""):
    # df = df.drop(["is_have_experience_lesson"],axis=1)
    import datetime
    if min_date == None:
        min_date = df['order_apply_time'].min()
    else:
        min_date = datetime.datetime.strptime(min_date, "%Y-%m-%d")

    if max_date == None:
        max_date = df['order_apply_time'].max()
    else:
        max_date = datetime.datetime.strptime(max_date, "%Y-%m-%d")


    df = df.drop("student_sex_fillna", axis=1)

    df_null = pd.DataFrame(df.isnull().sum())
    df_null.to_csv("load_data/df_null.csv")

    df["order_apply_time"] = pd.to_datetime(df["order_apply_time"])
    df["adjust_start_time"] = pd.to_datetime(df["adjust_start_time"])

    # 学生年级
    df['student_grade'] = df["exam_year"] - df["order_apply_time"].dt.year

    #排课日期相关月日时
    df["order_month"] = df["order_apply_time"].dt.month
    df["order_day"] = df["order_apply_time"].dt.day
    df["order_hour"] = df["order_apply_time"].dt.hour

    #上课日期相关月日时

    df["lesson_month"] = df["order_apply_time"].dt.month
    df["lesson_day"] = df["order_apply_time"].dt.day
    df["lesson_hour"] = df["order_apply_time"].dt.hour


    #销售特征
    df["sale_pigeon_rate"] = (df["sale_pigeon_counts"]+0.1)/(df["sale_total_counts"]+0.1)
    df["lm_sale_pigeon_rate"] = (df["lm_sale_pigeon_counts"]+0.1)/(df["lm_sale_total_counts"]+0.1)
    df["l3m_sale_pigeon_rate"] = (df["l3m_sale_pigeon_counts"] + 0.1)/(df["l3m_sale_total_counts"] + 0.1)
    df["his_sale_pigeon_rate"] = (df["his_sale_pigeon_counts"] + 0.1) / (df["his_sale_total_counts"] + 0.1)

    #意向特征
    df["commu_counts"]  = df["high_intention_counts"] + df["low_intention_counts"] + df["lost_connection_counts"]
    df["high_intenetion_rate"] = df["high_intention_counts"] / df["commu_counts"]
    df["low_intenetion_rate"] = df["low_intention_counts"] / df["commu_counts"]
    df["lost_intenetion_rate"] = df["lost_connection_counts"] / df["commu_counts"]


    # process student score
    def score_min(x):
        try:
            value = min(list(map(
                float,
                x.replace("分", "").replace("//", "/").replace("-", "/").split(
                    "/"))))
        except:
            value = np.nan
        return value

    def score_max(x):
        try:
            value = max(list(map(
                float,
                x.replace("分", "").replace("//", "/").replace("-", "/").split(
                    "/"))))
        except:
            value = np.nan
        return value

    df["score_min"] = round(df["recent_scores"].apply(score_min), 2)
    df["score_max"] = round(df["recent_scores"].apply(score_max), 2)
    df["score_mean"] = df["score_min"] / df["score_max"]
    df = df.fillna(0)
    df = df.dropna()

    select_columns = [
        "order_apply_time",

        # "sale_id",
        # "teacher_id",
        "is_pigeon",
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
        "exam_year",
        "coil_in",

        # "score_mean",
        # "self_evaluation_length",
        # "score_min",
        # "learning_target_lenght",
        # "score_max",
        #
        # "order_month",
        # "order_day",
        # "order_hour",
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

    # 数据预处理
    # df = df[select_columns]
    #
    # df = feature_onehot(df, label=label, features=["know_origin",
    #                                                 "grade_subject",
    #                                                 "student_city_class",
    #                                                 "student_province",
    #                                                 "grade_rank",
    #                                                 "class_rank_fillna",
    #                                                 "student_province_byphone",
    #                                                 "subject_ids",
    #                                                 "student_grade_lpo",
    #                                                 "school_background",
    #                                                 "is_first_trail",
    #                                                 "class_background_label",
    #                                                 'student_grade',
    #                                                 "exam_year",
    #                                                 "coil_in"], condition=1)

    df_train = df[(df["order_apply_time"] >= min_date) & (df["order_apply_time"]< mid_date)].drop(["order_apply_time"], axis=1)
    df_btest = df[(df["order_apply_time"] >= mid_date) & (df["order_apply_time"] <= max_date)].drop(["order_apply_time"], axis=1)
    print(min_date, mid_date, max_date)

    return df_train, df_btest


def data_clean2(df):
    #  最近六个月单笔金额
    df["p6m_avg_order_amt"] = df["CAFE20_AMT"] / (df["p6m_trans"] + 0.001)
    # food占比
    df["food_rate"] = df["CAFE20_VISIT_FOOD"] / (df["p6m_trans"] + 0.001)
    #  food_bev_rate
    df["bev_food_rate"] = df["CAFE20_VISIT_bev_food"] / (df["p6m_trans"] + 0.001)
    #  merche_rate
    df["merch_rate"] = df["cafe_tag_p6m_merch_qty"] / (df["p6m_trans"] + 0.001)
    # skr_rate
    df["skr_rate"] = df["CAFE20_VISIT_SRKIT"] / (df["p6m_trans"] + 0.001)

    df = df.drop(["member_id", 'partition', 'Unnamed: 0', 'p_date'], axis=1)
    df = df.fillna(0)

    # df = pd.get_dummies(df)
    cat_features = [
        "CAFE20_gender",
        "CAFE20_region",
        "CAFE20_levels"
    ]

    for catfeatures in cat_features:
        dict = {label: idx for idx, label in zip(range(len(df[catfeatures].unique())), df[catfeatures].unique())}
        df[catfeatures] = df[catfeatures].map(dict)

    df = df.astype(np.float32)



    # df = feature_onehot(df, label="target_is_DD_ACTIVE", features=cat_features, condition=1)

    df_train = df.copy()
    df_btest = df_train.sample(n=None, frac=0.1, replace=False, weights=None,
                               random_state=0, axis=0)
    return df_train, df_btest


def feature_onehot(df, label='issucess', features=['s', 'x'], condition=1):
    if condition == 1:
        df1 = df[features]#onehot对象
        df2 = df.drop(features, axis=1)#非0nehot对象
    else:
        features.append(label)
        df1 = df.drop(features, axis=1)
        df2 = df[features]
    for x in df1.columns:
        t = np.unique(df1[x])
        ts = []
        for i in t:
            vs = x + str(i)
            ts.append(vs)
        dd = pd.get_dummies(df1[x])
        dd.columns = ts
        df2 = pd.concat([df2, dd], axis=1)
    return df2


def seperate_label(df, label):
    x = df.drop(label, axis=1)
    y = df[label]
    return x, y


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
def data_seperate(df_train, df_btest,size=0.3, cri=None, undeal_column=['student_id'],label="is_sucess_by_contract"):
    X_train, X_test, y_train, y_test = train_test_split(df_train,df_train[label], test_size=size, random_state=0)
    if cri == None:
        X_train_tra = X_train.copy()
        X_test_tra = X_test.copy()
        df_btest_tra = df_btest.copy()

    elif cri == "log":
        X_train_tra = np.log(X_train+1)
        X_test_tra = np.log(X_test+1)
        df_btest_tra = np.log(df_btest+1)

    elif cri == "sigmod":
        X_train_tra = 1.0 / (1.0 + np.exp(-1*X_train)).reset_index(drop=True)
        X_test_tra = 1.0 / (1.0 + np.exp(-1*X_test)).reset_index(drop=True)
        df_btest_tra = 1.0 / (1.0 + np.exp(-1*df_btest)).reset_index(drop=True)

    elif cri == "sin":
        X_train_tra = np.sin(X_train)
        X_test_tra = np.sin(X_test)
        df_btest_tra = np.sin(df_btest)

    elif cri == "tanh":
        X_train_tra = np.tanh(X_train)
        X_test_tra = np.tanh(X_test)
        df_btest_tra = np.tanh(df_btest)


    else:
        if cri == 'standard':
            sc = StandardScaler()
        if cri == 'minmax':
            sc = MinMaxScaler()
        if cri == 'normal':
            sc = Normalizer()
        sc.fit(X_train)
        X_train_tra = pd.DataFrame(sc.transform(X_train))
        X_test_tra = pd.DataFrame(sc.transform(X_test))
        df_btest_tra = pd.DataFrame(sc.transform(df_btest))

        X_train_tra.columns = df_train.columns
        X_test_tra.columns = df_train.columns
        df_btest_tra.columns = df_train.columns

    X_train_tra = X_train_tra.reset_index(drop=True)
    X_test_tra = X_test_tra.reset_index(drop=True)
    df_btest_tra = df_btest_tra.reset_index(drop=True)

    if undeal_column==None:
        X_test_tra= X_test_tra
        X_test_tra = X_test_tra
        df_btest_tra = df_btest_tra

    else:
        for col in undeal_column:
            X_train_tra[col] = X_train.reset_index(drop=True)[col]
            X_test_tra[col] = X_test.reset_index(drop=True)[col]
            df_btest_tra[col] = df_btest.reset_index(drop=True)[col]

    return X_train_tra, X_test_tra, df_btest_tra
 


import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
from sklearn import cluster
from scipy.stats import itemfreq

def plot_eda(shuju, cut_number=10, fig_size=20, view=1, label='issucess',discrete_variable=['x','y'],method='cut',):
    # print(shuju.isnull().sum())
    save_dir = 'eda/image_plot_eda/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    writer = pd.ExcelWriter('eda/image_plot_eda/df_save.xlsx')
    coll = list(shuju.columns)
    list(np.unique(shuju.columns))
    coll.remove(label)

    # 方差表
    lens = len(np.unique(shuju[label]))
    sd = pd.DataFrame(np.zeros((len(coll), lens + 3)))
    sd.columns = ['feature', 'zero_std', 'one_std', 'sum_std', 'rate_std']

    k =0
    for x in coll:
        if x not in discrete_variable:
            if method=='cut':
                cut_factor = pd.cut(shuju[x],cut_number)
            if method=='qcut':
                cut_factor = pd.qcut(shuju[x],cut_number,duplicates='drop')
            group_df = shuju[label].groupby([cut_factor,shuju[label]]).count().unstack()
        if x in discrete_variable:
            group_df = shuju[label].groupby([shuju[x], shuju[label]]).count().unstack()
        group_df = group_df.fillna(0)
        group_df['sum'] = 0
        lab = np.unique(shuju[label])
        for h in list(lab):
            group_df['sum'] = group_df['sum'] + group_df[h]
        group_df['rate'] = group_df[view] / group_df['sum']

        sd.iloc[k, 0] = x
        sd.iloc[k, 1] = group_df[0].std()
        sd.iloc[k, 2] = group_df[1].std()
        sd.iloc[k, 3] = group_df['sum'].std()
        sd.iloc[k, 4] = group_df['rate'].std()

        k = k + 1
        # 堆积图
        plt.figure(figsize=(fig_size, fig_size))
        plt.xticks(fontsize=15,rotation=90)
        plt.yticks(fontsize=15)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.bar([str(x) for x in group_df.index], group_df.iloc[:, 0], label='0')
        plt.title('堆积图')
        for i in range(len(np.unique(shuju[label])) - 1):
            plt.bar([str(x) for x in group_df.index], group_df.iloc[:, i + 1], bottom=group_df.iloc[:, i], label='1')
        plt.legend()
        plt.xlabel(x, fontsize=18)
        plt.ylabel('数量')
        # 数据标签
        for a, b, c in zip([str(x) for x in group_df.index], group_df['rate'], group_df[1]):
            plt.text(a, c / 2, '%.2f' % b, ha='center', va='bottom', fontsize=20)
        path = save_dir
        paths = path + x + '.png'
        plt.savefig(paths)
        plt.show()
        group_df.to_excel(writer, sheet_name=x)
    writer.save()
    sd.to_excel('eda/sd_save.xlsx')


def feature_extend(df,extend_columns=[]):
    df_extend = df[extend_columns]
    # print("ori",len(df))
    #sd标准化
    sc1 = StandardScaler()
    sc1.fit(df_extend)
    df_sd = pd.DataFrame(sc1.transform(df_extend)).reset_index(drop=True)
    df_sd.to_csv("train_data/df.sd.csv",encoding="utf-8")
    sd_columns = []
    for col in df_extend.columns:
        colms = col + "_sd"
        sd_columns.append(colms)
    df_sd.columns = sd_columns
    #min-max标准化
    sc2 = MinMaxScaler()
    sc2.fit(df_extend)
    df_mm = pd.DataFrame(sc2.transform(df_extend)).reset_index(drop=True)
    mm_columns = []
    for col in df_extend.columns:
        colms = col + "_mm"
        mm_columns.append(colms)
    df_mm.columns = mm_columns

    #normalizer标准化
    sc3 = Normalizer()
    sc3.fit(df_extend)
    df_nor = pd.DataFrame(sc3.transform(df_extend)).reset_index(drop=True)
    nor_columns = []
    for col in df_extend.columns:
        colms = col + "_nor"
        nor_columns.append(colms)
    df_nor.columns = nor_columns

    #对数变换
    df_log = np.log(df_extend+1).reset_index(drop=True)
    log_columns = []
    for col in df_extend.columns:
        colms = col + "_log"
        log_columns.append(colms)
    df_log.columns = log_columns
    #指数变换
    df_exp=np.exp(df_extend/1000).reset_index(drop=True)
    exp_columns = []
    for col in df_extend.columns:
        colms = col + "_exp"
        exp_columns.append(colms)
    df_exp.columns = exp_columns

    #sin变换
    df_sin = np.sin(df_extend).reset_index(drop=True)
    sin_columns=[]
    for col in df_extend.columns:
        colms = col + "_sin"
        sin_columns.append(colms)
    df_sin.columns = sin_columns


    #sigmod变换
    df_sig = 1.0 / (1.0 + np.exp(-1*df_extend)).reset_index(drop=True)
    sig_columns = []
    for col in df_extend.columns:
        colms = col + "_sig"
        sig_columns.append(colms)
    df_sig.columns = sig_columns


    ds = pd.concat([df.reset_index(drop=True),
                    # df_sd,
                    # df_exp,
                    # df_log,
                    # df_mm,
                    # df_nor,
                    # df_sig,
                    df_sin
                    ],axis=1
                   )
    # print("last",len(ds))

    return ds
