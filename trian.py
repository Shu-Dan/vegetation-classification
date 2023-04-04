import os
import datetime
import matplotlib.pyplot as plt
import xlwt as xlwt

from keras.callbacks import  ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


from read import color_dict
from generator import trainGenerator,valGenerator
from deeplabv3 import Deeplabv3
from pspnet import pspnet
from vgg16unet import Unet
from unett import Unett
from Unet import unet
from atten_unett import Unet_plusplus
from unet1 import unet1
from UHRnet import HRnet


import tensorflow as tf

import sys  # 导入sys模块
sys.setrecursionlimit(3000)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)



train_image_path = r"G:\sj\dcnn_rf\sj\a\czt\JPEGImages900"
train_label_path = r"G:\sj\dcnn_rf\sj\a\czt\SegmentationClass900"


dataset_path=r"G:\sj\dcnn_rf\sj\a\czt"
traintxt_path="Segmentation/train.txt"
with open(os.path.join(dataset_path, "Segmentation/train.txt"), "r") as f:
    train_lines = f.readlines()
    # 打开数据集的txt
with open(os.path.join(dataset_path, "Segmentation/val.txt"), "r") as f:
    val_lines = f.readlines()

dice_loss=False

#  批大小
batch_size = 2
#  类的数目(包括背景)
classNum = 5
#  模型输入图像大小
input_size = (256, 256, 10)
#  训练模型的迭代总轮数
epochs = 150
#  初始学习率
learning_rate = 1e-4
#  预训练模型地址

premodel_path = None
#  训练模型保存地址
model_path = "Model\\vggunett1-gp.h5"
train_num = len(train_lines)
validation_num = len(val_lines)
steps_per_epoch = train_num / batch_size
validation_steps = validation_num / batch_size
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)


train_Generator = trainGenerator(batch_size,
                                 train_image_path,
                                 train_label_path,
                                 classNum,
                                 colorDict_GRAY,
                                 input_size)

validation_data = valGenerator(batch_size,
                                 train_image_path,
                                 train_label_path,
                                 classNum,
                                 colorDict_GRAY,
                                 input_size)

model = Unett(input_size=input_size,n_class=classNum, re_shape=False)


model_path1 = "F:/PycharmProjects/dnet/model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

model.load_weights(model_path1, by_name=True, skip_mismatch=True)


#  打印模型结构
model.summary()
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, verbose = 1)


#  回调函数
model_checkpoint = ModelCheckpoint(model_path,
                                   monitor='loss',
                                   verbose=1,  # 日志显示模式:0->安静模式,1->进度条,2->每轮一行
                                   save_best_only=True)

#  获取当前时间
start_time = datetime.datetime.now()
#  模型训练
history = model.fit_generator(train_Generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              callbacks=[model_checkpoint],
                              validation_data=validation_data,
                              validation_steps=validation_steps
                              )
#  训练总时间
end_time = datetime.datetime.now()
log_time = "训练总时间: " + str((end_time - start_time).seconds / 60) + "m"
print(log_time)
with open('TrainTime.txt', 'w') as f:
    f.write(log_time)

loss = history.history['loss']
val_loss = history.history['val_loss']
acc1 = history.history['l1_out_accuracy']
val_acc1 = history.history['val_l1_out_accuracy']
loss1 = history.history['l1_out_loss']
val_loss1 = history.history['val_l1_out_loss']
acc2 = history.history['l2_out_accuracy']
val_acc2 = history.history['val_l2_out_accuracy']
loss2 = history.history['l2_out_loss']
val_loss2 = history.history['val_l2_out_loss']
acc3 = history.history['l3_out_accuracy']
val_acc3 = history.history['val_l3_out_accuracy']
loss3 = history.history['l3_out_loss']
val_loss3 = history.history['val_l3_out_loss']
acc4 = history.history['l4_out_accuracy']
val_acc4 = history.history['val_l4_out_accuracy']
loss4 = history.history['l4_out_loss']
val_loss4 = history.history['val_l4_out_loss']
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('test', cell_overwrite_ok=True)
for i in range(len(loss)):
    sheet.write(i, 0, loss[i])
    sheet.write(i, 1, val_loss[i])
    sheet.write(i, 2, acc1[i])
    sheet.write(i, 3, val_acc1[i])
    sheet.write(i, 4, loss1[i])
    sheet.write(i, 5, val_loss1[i])
    sheet.write(i, 6, acc2[i])
    sheet.write(i, 7, val_acc2[i])
    sheet.write(i, 8, loss2[i])
    sheet.write(i, 9, val_loss2[i])
    sheet.write(i, 10, acc3[i])
    sheet.write(i, 11, val_acc3[i])
    sheet.write(i, 12, loss3[i])
    sheet.write(i, 13, val_loss3[i])
    sheet.write(i, 14, acc4[i])
    sheet.write(i, 15, val_acc4[i])
    sheet.write(i, 16, loss4[i])
    sheet.write(i, 17, val_loss4[i])
book.save(r'predict1118\AccAndLoss-qy5vggunett1-gp.xls')
epochs = range(1, len(loss) + 1)
plt.plot(epochs, acc1, 'r', label = 'Training acc1')
plt.plot(epochs, val_acc1, 'r', label = 'Validation acc1')
plt.plot(epochs, acc2, 'b', label = 'Training acc2')
plt.plot(epochs, val_acc2, 'b', label = 'Validation acc2')
plt.plot(epochs, acc3, 'g', label = 'Training acc3')
plt.plot(epochs, val_acc3, 'g', label = 'Validation acc3')
plt.plot(epochs, acc4, 'black', label = 'Training acc4')
plt.plot(epochs, val_acc4, 'black', label = 'Validation acc4')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("predict1118\\accuracy-qy5vggunett1-gp.png",dpi = 300)
plt.figure()
plt.plot(epochs, loss1, 'r', label = 'Training loss1')
plt.plot(epochs, val_loss1, 'r', label = 'Validation loss1')
plt.plot(epochs, loss2, 'b', label = 'Training loss2')
plt.plot(epochs, val_loss2, 'b', label = 'Validation loss2')
plt.plot(epochs, loss3, 'g', label = 'Training loss3')
plt.plot(epochs, val_loss3, 'g', label = 'Validation loss3')
plt.plot(epochs, loss4, 'black', label = 'Training loss4')
plt.plot(epochs, val_loss4, 'black', label = 'Validation loss4')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("predict1118\\loss-qy5vggunett1-gp.png", dpi = 300)
plt.show()