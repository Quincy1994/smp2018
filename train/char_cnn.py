# coding=utf-8

import os
import gc
#
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import pickle
import keras
import keras.backend as K

from init.config import Config
from my_utils.data_preprocess import to_categorical, char_cnn_train_batch_generator, get_char_seq
from model.deepzoo import get_textcnn
from my_utils.metrics import score

print("Load train @ test")
train = pd.read_csv(Config.train_data_path, sep='\t')
val = pd.read_csv(Config.test_data_path, sep='\t')
val_label = to_categorical(val.label)

load_val = True
batch_size = 64
model_name = "char_cnn"
trainable_layer = ["embedding"]
train_batch_generator = char_cnn_train_batch_generator

print("Load Val Data")
if load_val:
    val_char_seq = pickle.load(open(Config.cache_dir + '/g_val_char_seq_%s.pkl'%Config.char_seq_maxlen, "rb"))
else:
    val_char_seq = get_char_seq(val.content.values)
    gc.collect()
    pickle.dump(val_char_seq, open(Config.cache_dir + '/g_val_char_seq_%s.pkl'%Config.char_seq_maxlen, "wb"))
print(np.shape(val_char_seq))


print("Load Char embed")
char_embed_weight = pickle.load(open(Config.char_embed_path, "rb"))

model = get_textcnn(Config.char_seq_maxlen, char_embed_weight)
from keras.utils.vis_utils import plot_model
plot_model(model, to_file= model_name+ '.png',show_shapes=True)

for i in range(15):
    if i == 8:
        K.set_value(model.optimizer.lr, 0.0001)
    if i == 12:
        for l in trainable_layer:
            model.get_layer(l).trainable = True
    model.fit_generator(
        train_batch_generator(train.content.values, train.label.values, batch_size=batch_size),
        epochs=1,
        steps_per_epoch=int(train.shape[0]/batch_size),
        validation_data = (val_char_seq, val_label)
    )
    pred = np.squeeze(model.predict(val_char_seq))
    pre, rec, f1 = score(pred, val_label)
    print("precision", pre)
    print("recall", rec)
    print("f1_score", f1)
    model.save(Config.cache_dir + '/char_cnn/dp_embed_%s_epoch_%s_%s.h5' % (model_name, i, f1))