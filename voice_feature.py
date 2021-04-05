import pickle
from data_treatment import load_data_yf,data_clean,seperate_label,data_seperate,load_data_new,plot_eda,data_clean,feature_extend,data_clean2,feature_onehot
import pandas as pd
import jieba
import time
from tqdm import tqdm
import numpy as np


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
    labels = "attend"


    file = "load_data/audio_data_with_success_attend_cutwords.pkl"
    with open(file, 'rb') as f:
        data = pickle.load(f)

    #student_intention_id
    sql_student = """SELECT v.student_id,v.student_intention_id,v.student_no,lpo.apply_time AS order_apply_time,lpo.order_id ,v.submit_time from hfjydb.view_student v
    LEFT JOIN hfjydb.lesson_plan_order lpo on lpo.student_intention_id = v.student_intention_id"""

    df_student = load_data_new(sql_student, filename="df_students_s.csv")
    df_student = df_student[~df_student["order_id"].isnull()]

    df_concat = pd.merge(data, df_student, on="student_intention_id", how="left")
    df_concat = df_concat[~df_concat["order_id"].isnull()]

    conts = df_concat["student_intention_id"].value_counts()
    conts = conts[conts == 1]

    df_concat = df_concat[df_concat["student_intention_id"].isin(list(conts.index))]
    df_voice = df_concat[["order_id", "order_apply_time", "attend", "content"]]

    print(1)
    #分词
    start = time.clock()

    df_voice['words'] = [pd.Series(jieba.cut(x)) for x in tqdm(df_voice["content"])]
    print("用时:", time.clock() - start)

    start = time.clock()
    w = []  # 将所有词语整合在一起
    for i in df_voice["words"]:
        w.extend(i)
    dict = pd.DataFrame(pd.Series(w).value_counts())  # 统计词的出现次数
    # del w, d2v_train
    dict['id'] = list(range(1, len(dict) + 1))
    print(2)

    import datetime
    tqdm.pandas(desc="my bar！")
    get_sent = lambda x: list(dict['id'][x])
    df_voice['sent'] = df_voice['words'].progress_apply(get_sent) # 速度太慢

    print("用时:", time.clock() - start)

    # d = {}
    # for x in dict.index:
    #     d[x] = dict.loc[x, "id"]
    # df_voice["sents"] = [x.map(d) for x in tqdm(df_voice["words"])]
    # print("用时:", time.clock() - start)


    maxlen = 50

    print("Pad sequences (samples x time)")
    df_voice['sent'] = list(sequence.pad_sequences(df_voice['sent'], maxlen=maxlen))


    df_train = df_voice[(df_voice["order_apply_time"] >= "2018-09-01") & (
        df_voice["order_apply_time"] < "2019-01-10")].drop(["order_apply_time","content"], axis=1)
    df_btest = df_voice[(df_voice["order_apply_time"] >= "2019-01-10") & (
        df_voice["order_apply_time"] < "2019-01-15")].drop(["order_apply_time","content"], axis=1)


    # # #划分训练测试集
    X_train_tra, X_test_tra, df_btest = data_seperate(df_train, df_btest,
                                                      size=0.3,
                                                      label=labels,
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
    model.add(Embedding(input_dim=len(dict), output_dim=32, input_length=maxlen))
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


    x_trains["voice_prob"] = y_pred_train_prob
    x_tests["voice_prob"] = y_pred_test_prob
    x_btests["voice_prob"] = y_pred_btest_prob



    dense1_layer_model = Model(inputs=model.input, outputs = model.get_layer('Dense_1').output)

    dense1_output_train = dense1_layer_model.predict(x_train)
    dense1_output_test = dense1_layer_model.predict(x_test)
    dense1_output_btest = dense1_layer_model.predict(x_btest)

    x_trains["voice_dens1"] = dense1_output_train
    x_tests["voice_dens1"] = dense1_output_test
    x_btests["voice_dens1"] = dense1_output_btest

    result_df = pd.concat([x_trains, x_tests, x_btests], axis=0)
    result_df = result_df[["order_id", "voice_prob", "voice_dens1"]]
    result_df.to_csv("load_data/voice__prob_features.csv")

































