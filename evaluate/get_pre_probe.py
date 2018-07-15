# coding=utf-8

from init.config import Config

import pandas as pd
from my_utils.metrics import score
from my_utils.data_preprocess import to_categorical
from my_utils.data_preprocess import get_word_seq, get_char_seq
import pickle
import gc

import keras
from keras.models import *
import time

print("val")
val = pd.read_csv(Config.vali_data_path, sep='\t')
val_ids = val.id.values
val_content = val.content.values
val_word_seq = get_word_seq(val_content)
val_char_seq = get_char_seq(val_content)
val_seq = [val_word_seq, val_char_seq]
print("load ok")

print("test")
test = pd.read_csv(Config.test_data_path, sep='\t')
test_ids = test.id.values
test_labels = test.label.values
test_content = test.content.values
test_word_seq = get_word_seq(test_content)
test_char_seq = get_char_seq(test_content)
test_seq = [test_word_seq, test_char_seq]
print("load ok")

print("final")
final = pd.read_csv(Config.final_data_path, sep='\t')
final_ids = final.id.values
final_content = final.content.values
print("len of final_ids", len(final_ids))
print("len of final_content", len(final_content))
final_word_seq = get_word_seq(final_content)
final_char_seq = get_char_seq(final_content)
final_seq = [final_word_seq, final_char_seq]
print("load ok")


def predict_probe(model_name, model_list, type, seq):
    save_path = Config.cache_dir + '/final_probe_v2/%s_%s_'%(model_name, type)
    for i in range(len(model_list)):
        model_path = model_list[i]
        print(model_path)
        model = load_model(model_path)
        pre_pro = model.predict(seq)
        print(np.shape(pre_pro))
        pickle.dump(pre_pro, open(save_path+ str(i) + '.pk', 'wb'))
        del model
        gc.collect()

import datetime

start_time = datetime.datetime.now()
predict_probe(model_name="word_rcnn_char_rnn", model_list=Config.word_rcnn_char_rnn_list, type="final", seq=final_seq)
model_time = datetime.datetime.now()
