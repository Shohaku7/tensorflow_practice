import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt

# 定义过程可视化方法

def plot_training(history=None, metric='acc', title='Model Accuracy', loc='lower right'):
    model_list = []
    fig = plt.figure(figsize=(10, 8))
    for key, val in history.items():
        model_list.append(key.replace(HISTORY_DIR, '').rstrip('.history'))
        plt.plot(val[metric])

    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(model_list, loc=loc)
    plt.show()

# 加载预训练模型记录

HISTORY_DIR = './pre-trained/history/loss/'
history = {}
for filename in glob.glob(HISTORY_DIR + '*.history'):
    with open(filename, 'rb') as f:
        history[filename] = pickle.load(f)
for key, val in history.items():
    print(key.replace(HISTORY_DIR, '').rstrip('.history'), val.keys())
## 准确率变化（训练集）
# del history[HISTORY_DIR+'captcha_rmsprop_categorical_crossentropy_bs_100_epochs_100'+'.history']
plot_training(history)

## 损失值变化（训练集）
plot_training(history, metric='loss', title='Model Loss', loc='upper right')

## 准确率变化（测试集）

plot_training(history, metric='val_acc', title='Model Accuracy (val)')

## 损失值变化（测试集）


plot_training(history, metric='val_loss', title='Model Loss (val)', loc='upper right')