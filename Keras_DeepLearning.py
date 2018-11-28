# coding: utf-8
import os
import sys
from datetime import datetime

import matplotlib as plt
import numpy as np
import seaborn as sn
import tensorflow as tf

import random

import pandas as pd
from pandas import Series,DataFrame

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.optimizers import Adam

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping, CSVLogger

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support

import h5py

#GPU使用 #GPUを使用しない場合は、消す
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
config.gpu_options.visible_device_list="0"
sess = tf.Session(config=config)
K.set_session(sess)

########### accuracy, loss を描画する関数 ##########
def plot_results(H):

    loss = H.history['loss']
    val_loss = H.history['val_loss']
    acc = H.history['acc']
    val_acc = H.history['val_acc']

    epochs = len(acc)
    plt.pyplot.plot(range(epochs), acc, marker='.', label='acc')
    plt.pyplot.plot(range(epochs), val_acc, marker='.', label='val_acc')
    plt.pyplot.legend(loc='best')
    plt.pyplot.grid()
    plt.pyplot.xlabel('epoch')
    plt.pyplot.ylabel('acc')
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.pyplot.savefig(save_dir + data_name + '_accuracy_' + time)
    plt.pyplot.clf()
    epochs = len(loss)
    plt.pyplot.plot(range(epochs), loss, marker='.', label='loss')
    plt.pyplot.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.pyplot.legend(loc='best')
    plt.pyplot.grid()
    plt.pyplot.xlabel('epoch')
    plt.pyplot.ylabel('loss')
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.pyplot.savefig(save_dir + data_name + '_loss_' + time)
########### accuracy, loss を描画する関数 ############

########### CONFUSION MATRIX を描画する関数 ##########
def plot_confusion(true_labels, pred_labels):
    labels = sorted(list(set(true_labels)))
    data_confusion = confusion_matrix(true_labels, pred_labels, labels=labels)
    
    df_confusion = pd.DataFrame(data_confusion, index=labels, columns=labels)

    accuracy = accuracy_score(true_labels, pred_labels)
    p, r, f, s = precision_recall_fscore_support(true_labels, pred_labels, beta=0.5)
    pre_re_f = np.vstack((p,r,f))

    plt.pyplot.figure(figsize = (10,7))
    sn.heatmap(df_confusion, annot = True, fmt="d", cmap='Blues')
    
    plt.pyplot.xlabel('Predict Labels', fontsize=14)
    plt.pyplot.ylabel('True Labels', fontsize=14)
    plt.pyplot.title('Speaker Recognition : Confusion Matrix \n Accuracy : ' + str(accuracy), fontsize=20)

    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.pyplot.savefig(save_dir + data_name + '_confusion_matrix' + time)
    #csvファイルとして保存
    np.savetxt(save_dir + "accuracy_report_" + time + ".csv", pre_re_f, delimiter=',')
########### CONFUSION MATRIX を描画する関数 ##########

########### DNN をする関数 ###########
# 引数：　, 返り値：model
def DNN_model(l1_out, l2_out, l3_out, l4_out, l5_out, l1_drop, l2_drop, l3_drop, l4_drop, l5_drop):
    #Deep Neural Network
    model = Sequential()
    model.add(Dense(int(l1_out),input_dim=num_data,activation='relu'))
    model.add(Dropout(l1_drop))
    model.add(Dense(int(l2_out),activation='relu'))
    model.add(Dropout(l2_drop))
    model.add(Dense(int(l3_out),activation='relu'))
    model.add(Dropout(l3_drop))
    model.add(Dense(int(l4_out),activation='relu'))
    model.add(Dropout(l4_drop))
    model.add(Dense(int(l5_out),activation='relu'))
    model.add(Dropout(l5_drop))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    #Deep Neural Network

    # フィットモデル
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

    return model
########### DNN をする関数 ###########

########### model の train をする関数 ###########
def train_model(M, validation_split):
    plot_model(M, to_file='model1.png')
    es = EarlyStopping(monitor='val_loss', patience=2)
    csv_logger = CSVLogger('training.log')

    #各epochのtraining/validationのlossやaccuracyを記録する
    hist = M.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_split=validation_split,
                callbacks=[es, csv_logger],
                #steps_per_epoch=200,
                #validation_steps=200,
                shuffle=True)

    return hist
########### model の train をする関数 ###########

########### model の evaluate をする関数 ###########
def evaluate_model(M):
    score = M.evaluate(x_test, y_test, verbose=0)
    #print('test loss:', score[0])
    #print('test acc:', score[1])
    return score[0] 
########### model の evaluate をする関数 ###########


########### model の predict をする関数 ###########
def predict_model(M, hist):    
    predict = M.predict_classes(x_p, batch_size=128, verbose=0)

    # 結果のプロット
    plot_results(H=hist)
    plot_confusion(y_p, predict)
########### model の predict をする関数 ###########

########### model を評価する関数 ###########
def t_e_p_model(M, vs):
    hist = train_model(M=model,validation_split=vs)
    evaluate_model(M=model)
    predict_model(M=model, hist=hist)
########### model を評価する関数 ###########

########### PARAMETERS ##########
#特徴量数（入力層の数）
num_data = 512
#出力層数
num_classes = 5

#バッチサイズとエポック
batch_size = 128
epochs = 20
########### PARAMETERS ##########

########### 画像の保存場所 ###########
save_dir = "C:/Users/robot/OneDrive/Desktop/Laboratory/Python/Keras_DeepLearning/fig"
os.makedirs(save_dir, exist_ok = True)
########### 画像の保存場所 ###########

######################### TRAIN DATASET #########################
#train_dataset_dir = "E:/dataset/" + directory_name + "/" + data_name + "/Analysis/" + data_name + option + "_train" + ".csv"
#SR_train_dataset = pd.read_csv(train_dataset_dir,sep=',',header=None)
train_dataset_dir = "C:/Users/robot/OneDrive/Desktop/Laboratory/Python/Keras_DeepLearning/Analysis/train.mat"
f_t = h5py.File(train_dataset_dir, 'r')
######################### TRAIN DATASET #########################

######################### PREDICT DATASET #########################
#predict_dataset_dir = "E:/dataset/" + directory_name + "/" + data_name + "/Analysis/" + data_name + option + "_predict" + ".csv"
#SR_predict_dataset = pd.read_csv(predict_dataset_dir,sep=',',header=None)
predict_dataset_dir = "C:/Users/robot/OneDrive/Desktop/Laboratory/Python/Keras_DeepLearning/Analysis/predict.mat"
f_p = h5py.File(predict_dataset_dir, 'r')
######################### PREDICT DATASET #########################

######################### TRAIN DATASET を x, y に読み込む#####################
# 引数labelsとaxisで指定する。行の場合はaxis=0。列の場合はaxis=1
x_t = DataFrame(f_t["RESULT"].value).drop(num_data,axis=1) #SR_datasetの特徴量だけxに入力。512列目以外を抽出
y_t = DataFrame(f_t["RESULT"].value)[:][num_data] #SR_datasetのラベルだけyに入力
######################### TRAIN DATASET を x, y に読み込む#####################

######################### PREDICT DATASET を x, y に読み込む#####################
# 引数labelsとaxisで指定する。行の場合はaxis=0。列の場合はaxis=1
x_p = DataFrame(f_p["RESULT"].value).drop(num_data,axis=1) #SR_datasetの特徴量だけxに入力。512列目以外を抽出
y_p = DataFrame(f_p["RESULT"].value)[:][num_data] #SR_datasetのラベルだけyに入力
print(x_p)
print(y_p)
######################### PREDICT DATASET を x, y に読み込む#####################

######################### TRAIN DATASET を x_train, x_test, y_train, y_test に分割 #####################
#特徴量とラベルをそれぞれ訓練、テストデータに分割
x_train,x_test,y_train,y_test = train_test_split(x_t,y_t,test_size=0.1)
######################### TRAIN DATASET を x_train, x_test, y_train, y_test に分割 #####################

######################### TRAIN DATASET の データ整形 #########################
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)
y_train = y_train.astype(np.int)
y_test = y_test.astype(np.int)
######################### TRAIN DATASET の データ整形 #########################

######################### PREDICT DATASET の データ整形 #######################
x_p = x_p.astype(np.float)
y_p = np.array(y_p.astype(np.int)).T.tolist()
######################### PREDICT DATASET の データ整形 #######################


################ 普通の学習 #################
"""
ln_out  : 中間層ｎ個目のニューロンの数
ln_drop : 中間層ｎ個目のドロップアウト率
vs      : train データから、何割を検証データとして用いるか。逆伝搬法実施のために必要

"""
model = DNN_model(l1_out = 1024,
                l2_out = 512,
                l3_out = 256,
                l4_out = 512,
                l5_out = 128,
                l1_drop = 0.01,
                l2_drop = 0.2,
                l3_drop = 0.3,
                l4_drop = 0.3,
                l5_drop = 0.3,
                )
t_e_p_model(M=model,
            vs=0.2
            )
################ 普通の学習 #################
