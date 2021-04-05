from data_treatment import load_data_new, data_seperate, seperate_label
import pandas as pd  # 导入Pandas
import numpy as np  # 导入Numpy
import jieba  # 导入结巴分词

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution1D

from model_evalu import evalution_model


if __name__ == '__main__':
    labels = "is_pigeon"

    sql = "SELECT * from bidata.trail_pigeon_wdf1"
    df = load_data_new(sql, filename="df_20190226.csv")

    student_sql = """SELECT student_id,count(student_id) from trail_pigeon_wdf1 
                            GROUP BY student_id"""
    student_mul = load_data_new(student_sql, filename="student_mul.csv")
    student_ids = student_mul[student_mul["count(student_id)"] == 1]

    df = df[df["student_id"].isin(student_mul["student_id"])]
    df = df[["order_id", "order_apply_time", "communicition_contents", "is_pigeon"]]

    import re
    df = df[(df["order_apply_time"] >= "2018-09-01") & (df["order_apply_time"] < "2019-01-15")]
    df = df.dropna()
    quoto = '\d{1,}|[’!#$%&\()+,、.-:;<=>?@[ \\\\ \]^_`{}~，、。：【】；”“？（）{ }{}\n\s\t/]|[a-z]|[A-Z]'
    print("start_sub")
    df["communicition_contents"] = [re.sub(quoto, '', x) for x in df["communicition_contents"]]
    df = df[df["communicition_contents"] != ""]
    print("end_sub")
    print(1)

    # cw = lambda x: list(jieba.cut(x))  # 定义分词函数
    # df['words'] = df["communicition_contents"].apply(cw)
    df['words'] = [list(jieba.cut(x)) for x in df["communicition_contents"]]

    print(3)
    w = []  # 将所有词语整合在一起
    for i in df["words"]:
        w.extend(i)
    print(2)
    dict = pd.DataFrame(pd.Series(w).value_counts())  # 统计词的出现次数
    # del w, d2v_train
    dict['id'] = list(range(1, len(dict) + 1))

    d = {}
    for x in dict.index:
        d[x] = dict.loc[x, "id"]

    import datetime
    print(datetime.datetime.now())
    get_sent = lambda x: list(dict['id'][x])
    df['sent'] = df['words'].apply(get_sent)  # 速度太慢

    # df["word"] = [pd.Series(x) for x in df["words"]]
    # print(datetime.datetime.now())
    #
    # i = 0
    # for x in df["word"]:
    #     x.map(d)
    #     print(i)
    #     i = i+1
    #
    # df["sents"] = [x.map(d) for x in df["word"]]
    print(datetime.datetime.now())

    print(4)
    maxlen = 50

    print("Pad sequences (samples x time)")
    df['sent'] = list(sequence.pad_sequences(df['sent'], maxlen=maxlen))

    df_train = df[(df["order_apply_time"] >= "2018-09-01") & (
        df["order_apply_time"] < "2019-01-10")].drop(["order_apply_time","communicition_contents"], axis=1)
    df_btest = df[(df["order_apply_time"] >= "2019-01-10") & (
        df["order_apply_time"] < "2019-01-15")].drop(["order_apply_time","communicition_contents"], axis=1)


    # # #划分训练测试集
    X_train_tra, X_test_tra, df_btest = data_seperate(df_train, df_btest,
                                                      size=0.3,
                                                      label="is_pigeon",
                                                      cri=None,
                                                      undeal_column=None
                                                      )

    # 划分label
    x_trains, y_train = seperate_label(X_train_tra, label=labels)
    x_tests, y_test = seperate_label(X_test_tra, label=labels)
    x_btests, y_btest = seperate_label(df_btest, label=labels)

    x_train = np.array(list(x_trains['sent']))
    x_test = np.array(list(x_tests['sent']))
    x_btest = np.array(list(x_btests['sent']))


    print("x_train", x_train.shape)

    model = Sequential()
    model.add(Embedding(input_dim=len(dict) + 1, output_dim=32, input_length=maxlen))
    # model.add(
        # Embedding(input_dim=128, output_dim=64, input_length=maxlen))/

    model.add(LSTM(128))  # try using a GRU instead, for fun
    model.add(Dropout(0.5))
    # model.add(Convolution1D())
    model.add(Dense(1, name="Dense_1"))

    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=256, nb_epoch=10)  # 训练时间为若干个小时
    # model.save('models/deep_model/lstm_0228.h5')
    # model = load_model('models/deep_model/lstm_0228.h5')

    # classes = model.predict_classes(xt)
    # acc = np_utils.accuracy(classes, yt)
    # score, acc = model.evaluate(xt, yt, batch_size=16, verbose=1)
    score, acc = model.evaluate(x_test, y_test, verbose=1)
    print('Test score:', score)
    print('Test accuracy:', acc)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred_btest = model.predict(x_btest)

    print("================训练集================")
    evalution_model(model, x_train, y_train)
    print("================测试集==============")
    evalution_model(model, x_test, y_test)
    print("================B测试集==============")
    evalution_model(model, x_btest, y_btest)

    y_pred_train_prob = model.predict_proba(x_train)
    y_pred_test_prob = model.predict_proba(x_test)
    y_pred_btest_prob = model.predict_proba(x_btest)


    x_trains["cnn_prob"] = y_pred_train_prob
    x_tests["cnn_prob"] = y_pred_test_prob
    x_btests["cnn_prob"] = y_pred_btest_prob



    dense1_layer_model = Model(inputs=model.input, outputs = model.get_layer('Dense_1').output)

    dense1_output_train = dense1_layer_model.predict(x_train)
    dense1_output_test = dense1_layer_model.predict(x_test)
    dense1_output_btest = dense1_layer_model.predict(x_btest)

    x_trains["cnn_dens1"] = dense1_output_train
    x_tests["cnn_dens1"] = dense1_output_test
    x_btests["cnn_dens1"] = dense1_output_btest

    result_df = pd.concat([x_trains, x_tests, x_btests], axis=0)
    result_df = result_df[["order_id", "cnn_prob", "cnn_dens1"]]
    result_df.to_csv("load_data/cnn_features_1.csv")









    






