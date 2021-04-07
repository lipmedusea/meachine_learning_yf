import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report


def plot_importance(clf, features, title='feature_importancet', n=5, method=None):
    save_dir = 'image/image_save/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    model_save_path = 'model_saved/'
    if os.path.exists(model_save_path) is False:
        os.makedirs(model_save_path)

    if method == "rf":
        importances = pd.DataFrame(clf.feature_importances_)
    if method == "lgb":
        importances = pd.DataFrame(clf.feature_importance(importance_type='split'))
    else:
        importances = pd.DataFrame(clf.best_estimator_.feature_importances_)
    feat_labels = pd.DataFrame(features)
    fs = pd.concat([feat_labels, importances], axis=1)
    fs.columns = ['features', 'importance']
    fd = fs.sort_values(by=["importance"], ascending=False).round(3)
    plt.figure(figsize=(15, 15))
    plt.bar(fd['features'][0:n], fd['importance'][0:n])
    plt.title(title)
    # plt.legend()
    path = 'image/image_save/importance.png'
    plt.savefig(path)
    plt.show()
    print(fd.head(n))


def evalution_model(clf, x, y, pic_name='trian'):
    y_preds = clf.predict(x)

    y_preds = np.where(y_preds > 0.5, 1, 0)
    #精度
    print(classification_report(y, y_preds, target_names=['0', '1']))
    # print('Precision: %.3f' % precision_score(y_true=y.astype(int), y_pred=y_preds.astype(int)))
    # print('Recall: %.3f' % recall_score(y_true=y.astype(int), y_pred=y_preds.astype(int)))
    # print('F1: %.3f' % f1_score(y_true=y.astype(int), y_pred=y_preds.astype(int)))
    #混淆矩阵
    confmat = confusion_matrix(y_true=y.astype(int), y_pred=y_preds.astype(int))
    print(confmat)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.tight_layout()
    path_confmat = 'image/image_save/confmat' + pic_name
    plt.savefig(path_confmat, dpi=300)
    plt.show()
    #ROC曲线
    fpr, tpr, _ = roc_curve(y, y_preds)  # 计算真正率和假正率
    roc_auc = auc(fpr, tpr)  # 计算auc的值
    print("AUC={0}".format(roc_auc))

    plt.figure(figsize=(10, 10))
    # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    path_roc = 'image/image_save/ROC' + pic_name
    plt.savefig(path_roc, dpi=300)
    plt.show()

from base_function import combine
import pandas as pd
import io
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
def comparation_train_test(clf, x_train, y_train, x_test, y_test, param_grid, score='recall',sort_by=['pre_zero','recall_zero']):
    import datetime
    start_time=datetime.datetime.now()
    #关键参数
    key_parameter = []
    para = param_grid.copy()
    for x in param_grid:
        key_parameter.append(x)
    #生成df
    para_df = combine(param_grid)
    colm = key_parameter.copy()
    list_index = ['pre_zero', 'recall_zero', 'f1_zero', 'pre_one', 'recall_one', 'f1_one',
              'pre_avg', 'recall_avg', 'f1_at avg']
    for d in list_index:
        colm.append(d)
    df = pd.DataFrame(columns=colm, index=range(len(para_df)))
    #para_df参数赋给df
    for g in para_df.columns:
        df[g] = para_df[g]
    k=1
    for i in range(len(para_df)):
        #para传入参数
        for y in key_parameter:
            tps = []
            tps.append(para_df.loc[i, y])
            para[y] = tps
        #模型训练
        clfs = GridSearchCV(estimator=clf,
                            param_grid=para,
                            scoring=score,
                            cv=5,
                            n_jobs=-1)
        clfs = clfs.fit(x_train, y_train.astype(int))
        #预测结果
        y_test_pred = clfs.predict(x_test)
        com_test = classification_report(y_test, y_test_pred, target_names=['0', '1'])
        df_test = pd.read_csv(io.StringIO(com_test.replace("avg / total", "avg/total")), sep="\s+").round(
            {"precision": 2, "recall": 2, "f1-score": 2, "support": 2})
        df_test = df_test.drop('support', axis = 1 )
        df_test = df_test.values.flatten()

        for z in range(len(list_index)):
            df.iloc[i, len(key_parameter):] = df_test
        end_time=datetime.datetime.now()
        print('step'+str(k),end_time-start_time)
        k = k+1
    df = df.sort_values(by=sort_by,ascending=[False,False])
    return df

import datetime
from collections import Counter
def mul_model(clf, x_train, y_train, x_test, y_test, para_df, score='recall'):
    cout = Counter(y_train)
    tt = cout[0] / cout[1]
    sample_weigh = np.where(y_train == 0, 1, tt)
    start_time = datetime.datetime.now()
    para_dict = dict(zip(para_df.index, range(len(para_df.index))))
    for x in para_dict:
        para_dict[x] = [para_df[x]]

    colum = list(para_df.index)
    list_index = ['pre_zero', 'recall_zero', 'f1_zero', 'pre_one', 'recall_one', 'f1_one',
                  'pre_avg', 'recall_avg', 'f1_at avg']
    for col in list_index:
        colum.append(col)

    df = pd.DataFrame(columns=colum, index=range(1))

    for g in para_df.index:
        df[g] = [para_df[g]]

    clfs = GridSearchCV(estimator=clf,
                        param_grid=para_dict,
                        scoring=score,
                        cv=5,
                        n_jobs=1)
    clfs = clfs.fit(x_train, y_train)
                    # sample_weight=sample_weigh)
    # 预测结果
    y_test_pred = clfs.predict(x_test)
    com_test = classification_report(y_test, y_test_pred, target_names=['0', '1'])
    df_test = pd.read_csv(io.StringIO(com_test.replace("avg / total", "avg/total")), sep="\s+").round(
        {"precision": 2, "recall": 2, "f1-score": 2, "support": 2})
    df_test = df_test.drop('support', axis=1)
    df_test = df_test.values.flatten()

    df.iloc[0, len(para_df.index):] = df_test
    endtime = datetime.datetime.now()
    print(endtime - start_time)
    return df


from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]

        self.indices_ = list(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                print(p)
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, list(p))
                scores.append(score)
                subsets.append(p)


            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X.iloc[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train.iloc[:, indices], y_train)
        y_pred = self.estimator.predict(X_test.iloc[:, indices])
        score = self.scoring(y_test, y_pred, pos_label=1)
        return score



def sort_by_model_intersection(*args, key = 'student_no', view=0):
    args = list(args)
    s=0
    for df in args:
        args[s] = df[df[view]>0.5]
        s=s+1

    #获取所有df下的key
    student_no_list = []
    pred_prob_column=[]
    i=1
    for df in args:
        student_no_list=student_no_list + list(df[key])
        dicts = {view: "df" + str(i) + "pred_" + str(view)}
        pred_prob_column.append(dicts[view])
        args[i-1]=args[i-1].rename(columns = dicts)
        i = i+1
    student_no_list =np.unique(student_no_list)

    #判断是否出现在dfi
    column = [key]
    for x in range(len(args)):
        column.append("isin_df"+str(x+1))

    # 生成空表
    df_sort = pd.DataFrame(columns=column)
    df_sort[key] = student_no_list

    #merge
    j = 1
    for df in args:
        dicts = {view: "df" + str(j) + "pred_" + str(view)}
        df_sort = pd.merge(df_sort,df[[key,dicts[view]]],how="left")
        df_sort["isin_df"+str(j)] = np.where(df_sort[dicts[view]].isnull(),0,1)
        j = j+1

    #计算出现次数与平均prob值
    df_sort['avg_prob'] = df_sort[pred_prob_column].mean(1)
    df_sort['appear_count'] = df_sort[column[1:]].sum(1)
    #排序
    df_sort= df_sort.sort_values(by=['appear_count','avg_prob'], ascending=False)
    df_sort['is_sent'] = np.where(df_sort['appear_count']>1,1,0)

    return df_sort

