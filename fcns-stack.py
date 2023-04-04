import os
import datetime

import math
from osgeo import gdal
import numpy as np
import cv2
from tensorflow.keras.models import Model
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn import model_selection
import pickle
import sklearn.metrics as sm
from sklearn import neighbors
from sklearn.svm import SVC
import tensorflow as tf
import pandas as pd
from saixuan import Relief
from keras import backend as K

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

def readTif1(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    return GdalImg_data

def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj

def Generator(train_image_path):
    TifArrayReturn = []

    dataset_path = r"G:\sj\cnn-rf\jg\gpsd\yz"
    txt_path = "Segmentation2/val.txt"
    f = open(os.path.join(dataset_path, txt_path), "r")
    train_lines = f.readlines()
    imageList = []
    for line in train_lines:
        temp1 = line.strip('\n')
        imageList.append(temp1 + ".tif")

    for i in imageList:
        img = readTif1(train_image_path + "\\" + i)
        m = tf.math.l2_normalize(img, dim=1)
        img = m.numpy()

        img = img.swapaxes(1, 0)
        img = img.swapaxes(1, 2)
        # img=img / 255.0
        TifArrayReturn.append(img)
    return TifArrayReturn

def Generator1(train_image_path):
    TifArrayReturn = []
    dataset_path = r"G:\sj\cnn-rf\jg\gpsd\yz"
    txt_path = "Segmentation2/val.txt"
    f = open(os.path.join(dataset_path, txt_path), "r")
    train_lines = f.readlines()
    imageList = []
    for line in train_lines:
        temp1 = line.strip('\n')
        imageList.append(temp1 + ".tif")

    for i in imageList:
        img = readTif1(train_image_path + "\\" + i).astype(np.uint8)
        if (len(img.shape) == 3):
            img = img.swapaxes(1, 0)
            img = img.swapaxes(1, 2)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #  GDAL读数据是(BandNum,Width,Height)要转换为->(Width,Height,BandNum)
        TifArrayReturn.append(img)
    return TifArrayReturn

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def Kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    return (po - pe) / (1 - pe)

def OverallAccuracy(confusionMatrix):
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA

area_perc = 0.5

TifPath = r"E:\shuju\fl\wsc_yczt20\a\bq\wsb1_11.tif"
ModelPath = r"Model\q10_net64w.h5"
laPath = r"E:\shuju\fl\wsc_yczt20\a\bq\f1.tif"

train_image_path = r"G:\sj\cnn-rf\a\JPEGImages12"
train_label_path= r"G:\sj\cnn-rf\a\SegmentationClass12"


#  记录测试消耗时间
testtime = []
#  获取当前时间
starttime = datetime.datetime.now()

TifArray1=Generator(train_image_path)
laArray1=Generator1(train_label_path)


model_path1 = "Model\\vggunett-gp.h5"
model_path2 = "Model\\unett-gp.h5"
model_path3 = "Model\\resnett-gp.h5"

model1=load_model(model_path1,custom_objects={'dice_coef': dice_coef,'bce_dice_loss':bce_dice_loss})
model2=load_model(model_path2,custom_objects={'dice_coef': dice_coef,'bce_dice_loss':bce_dice_loss})
model3=load_model(model_path3,custom_objects={'dice_coef': dice_coef,'bce_dice_loss':bce_dice_loss})


activation_model1 = Model(inputs=model1.input, outputs=model1.layers[-9].output)
activation_model2 = Model(inputs=model2.input, outputs=model2.layers[-9].output)
activation_model3 = Model(inputs=model3.input, outputs=model3.layers[-9].output)
model2.summary()

imge=np.array(TifArray1)
lable=np.array(laArray1)
features1 = activation_model1.predict(imge)
features2 = activation_model2.predict(imge)
features3 = activation_model3.predict(imge)
x1=features1.reshape(-1,features1.shape[3])
x2=features2.reshape(-1,features2.shape[3])
x3=features3.reshape(-1,features3.shape[3])
x4=np.insert(x1, [32], x2, axis=1)
x5=np.insert(x4, [64], x3, axis=1)
x=x1


m=list(range(96))
61, 18, 86, 66, 57, 27, 34, 49, 45, 80, 2, 77, 54, 26, 9, 67, 39, 35, 73, 15, 8, 37, 93, 81, 0, 16, 12, 5, 95, 69, 62, 25, 88, 43, 30, 7, 76, 90, 74, 56, 21, 4, 19, 87, 24, 29, 3, 53, 71, 44, 14, 10, 75, 68]
c=[63, 38, 46, 52, 37, 53, 44, 45, 41, 33, 36, 50, 58, 61, 59, 20, 49, 14, 47, 34, 17, 51, 48, 56, 32, 54, 43, 9, 19, 40, 6, 39, 42, 12, 60, 5, 55]
h= [x for x in m if x not in c]
x=np.delete(x5, h, axis=1)

y=lable.reshape(-1)


train_data, test_data, train_label, test_label = model_selection.train_test_split(x, y, random_state=1,
                                                                                  train_size=0.5, test_size=0.5)
train_data, test_data, train_label, test_label = model_selection.train_test_split(test_data, test_label, random_state=1,
                                                                                  train_size=0.8, test_size=0.2)


clf1 = KNeighborsClassifier(n_neighbors=1)
clf4 = LGBMClassifier(num_leaves=63)
clf3 = SVC(C=1.0, kernel='rbf',degree=6, cache_size=1024,probability=True)
lr = RandomForestClassifier(n_estimators=100, random_state=42)
sclf = StackingClassifier(classifiers=[ clf1, clf4,clf3],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=lr)



print('3-fold cross validation:\n')

for clf, label in zip([clf2,clf4, clf6, sclf],
                      [
                       'Random Forest',
                       'LGB',
                       'Boosting',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, train_data, train_label,
                                              cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
#验证集
sclf.fit(train_data, train_label)
pred_test_y = sclf.predict(test_data)
print("准确率",accuracy_score(test_label, pred_test_y))
cm = sm.confusion_matrix(test_label, pred_test_y)
print(cm)
cr = sm.classification_report(test_label, pred_test_y)
print(cr)
print(Kappa(cm))
print(OverallAccuracy(cm))


endtime = datetime.datetime.now()
text = "结果耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)
time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')