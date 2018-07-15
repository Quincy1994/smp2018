# coding=utf-8

from init.config import Config

import pandas as pd
from my_utils.metrics import score
from my_utils.data_preprocess import to_categorical
from my_utils.data_preprocess import get_word_seq,get_char_seq
import pickle
import gc
import time

import keras
from keras.models import *


def to_label(pred):
    label_dict = {0: "自动摘要", 1:"机器翻译", 2:"机器作者", 3:"人类作者"}
    result = []
    for i in range(len(pred)):
        result.append(label_dict[pred[i]])
    return result


def submit_result(ids, labels, result_path):
    length = len(ids)
    f = open(result_path, 'w')
    for i in range(length):
        line = str(ids[i]) + "," + str(labels[i]) + '\n'
        f.write(line)
    f.close()
    print("done!")


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
test_labels = to_categorical(test_labels)
test_content = test.content.values
test_word_seq = get_word_seq(test_content)
test_char_seq = get_char_seq(test_content)
test_seq = [test_word_seq, test_char_seq]
print("load ok")





def predict_probe(model_name, model_list, type, seq):
    save_path = Config.cache_dir + '/probe/%s_%s_'%(model_name, type)
    for i in range(len(model_list)):
        model_path = model_list[i]
        print(model_path)
        model = load_model(model_path)
        pre_pro = model.predict(seq)
        pickle.dump(pre_pro, open(save_path+str(i)+'.pk', 'wb'))
        print(pre_pro)
        print(np.shape(pre_pro))
        gc.collect()
        time.sleep(3)



def evaluate_test():
    test = pd.read_csv(Config.test_data_path, sep='\t')
    test_labels = test.label.values
    test_labels = to_categorical(test_labels)
    test_pro_cnn = pickle.load(open(Config.cache_dir + '/probe/cnn/test/2.pk', 'rb'))
    test_pro_rnn = pickle.load(open(Config.cache_dir + '/probe/rnn/test/2.pk', 'rb'))
    test_pro_rcnn = pickle.load(open(Config.cache_dir + '/probe/rcnn/test/1.pk', 'rb'))
    test_pro_deep_cnn = pickle.load(open(Config.cache_dir+'/probe/deep_cnn/test/0.pk', 'rb'))
    test_pro_word_rcnn_char_rnn = pickle.load(open(Config.cache_dir + '/probe/word_rcnn_char_rnn/test/2.pk', 'rb'))
    test_pro_word_rnn_char_rcnn = pickle.load(open(Config.cache_dir + '/probe/word_rnn_char_rcnn/test/2.pk', 'rb'))
    test_pro_word_char_cgru = pickle.load(open(Config.cache_dir + '/probe/word_char_cgru/test/1.pk', 'rb'))
    test_pro_word_rcnn_char_cgru = pickle.load(open(Config.cache_dir + '/probe/word_rcnn_char_cgru/test/1.pk', 'rb')) # best: 0
    test_pro_word_cgru_char_rnn = pickle.load(open(Config.cache_dir + '/probe/word_cgru_char_rnn/test/0.pk', 'rb'))
    test_pro_word_cgru_char_rcnn = pickle.load(open(Config.cache_dir + '/probe/word_cgru_char_rcnn/test/0.pk', 'rb'))
    test_pro_word_rnn_char_cgru = pickle.load(open(Config.cache_dir + '/probe/md/word_rnn_char_cgru/test/1.pk', 'rb'))
    test_pro_word_rnn_char_cnn = pickle.load(open(Config.cache_dir + '/probe/md/word_rnn_char_cnn/test/0.pk', 'rb'))

    test_pro = test_pro_cnn
    test_pro += test_pro_rnn
    test_pro += test_pro_rcnn
    test_pro += test_pro_deep_cnn
    test_pro += test_pro_word_rcnn_char_rnn
    test_pro += test_pro_word_rnn_char_rcnn
    test_pro += test_pro_word_char_cgru
    test_pro += test_pro_word_cgru_char_rcnn
    test_pro += test_pro_word_rcnn_char_cgru
    test_pro += test_pro_word_cgru_char_rnn
    test_pro += test_pro_word_rnn_char_cgru
    test_pro += test_pro_word_rnn_char_cnn

    pre, rec, f = score(test_pro, test_labels)
    print(pre)
    print(rec)
    print(f)
    print(np.mean(f))


def combine_all(model_name, type):
    pro = pickle.load(open(Config.cache_dir + '/probe/%s/%s/0.pk'%(model_name, type), 'rb'))
    pro += pickle.load(open(Config.cache_dir + '/probe/%s/%s/1.pk' % (model_name, type), 'rb'))
    pro += pickle.load(open(Config.cache_dir + '/probe/%s/%s/2.pk' % (model_name, type), 'rb'))
    return pro

def evaluate_test_all():
    test = pd.read_csv(Config.test_data_path, sep='\t')
    test_labels = test.label.values
    test_labels = to_categorical(test_labels)
    test_pro_cnn = combine_all('cnn', 'test')
    test_pro_rnn = combine_all('rnn', 'test')
    test_pro_rcnn = combine_all('rcnn', 'test')
    test_pro_deep_cnn = combine_all('deep_cnn', 'test')
    test_pro_word_rcnn_char_rnn = combine_all('word_rcnn_char_rnn', 'test')
    test_pro_word_rnn_char_rcnn = combine_all('word_rnn_char_rcnn', 'test')
    test_pro_word_char_cgru = combine_all('word_char_cgru', 'test')
    test_pro_word_rcnn_char_cgru = combine_all('word_rcnn_char_cgru', 'test')  # best: 0
    test_pro_word_cgru_char_rnn = pickle.load(open(Config.cache_dir + '/probe/word_cgru_char_rnn/test/0.pk', 'rb'))
    test_pro_word_cgru_char_rcnn = combine_all('word_cgru_char_rcnn', 'test')
    test_pro_word_rnn_char_cgru = combine_all('word_rnn_char_cgru', 'test')
    test_pro_word_rnn_char_cnn = combine_all('word_rnn_char_cnn', 'test')

    test_pro =  test_pro_cnn
    test_pro += test_pro_rnn
    test_pro += test_pro_rcnn
    test_pro += test_pro_deep_cnn
    test_pro += test_pro_word_rcnn_char_rnn
    test_pro += test_pro_word_rnn_char_rcnn
    test_pro += test_pro_word_char_cgru
    test_pro += test_pro_word_cgru_char_rcnn
    test_pro += test_pro_word_rcnn_char_cgru
    test_pro += test_pro_word_cgru_char_rnn
    test_pro += test_pro_word_rnn_char_cgru
    test_pro += test_pro_word_rnn_char_cnn

    pre, rec, f = score(test_pro, test_labels)
    print(pre)
    print(rec)
    print(f)
    print(np.mean(f))


def load_fasttext(ft_path):
    pro = pickle.load(open(ft_path, 'rb'), encoding='iso-8859-1')
    return pro



def ensemble_valid():
    val = pd.read_csv(Config.vali_data_path, sep='\t')
    val_ids = val.id.values
    val_pro_cnn = pickle.load(open(Config.cache_dir + '/probe/cnn/valid/2.pk', 'rb'))
    val_pro_rnn = pickle.load(open(Config.cache_dir + '/probe/rnn/valid/2.pk', 'rb'))
    val_pro_rcnn = pickle.load(open(Config.cache_dir + '/probe/rcnn/valid/1.pk', 'rb'))
    val_pro_deep_cnn = pickle.load(open(Config.cache_dir + '/probe/deep_cnn/valid/0.pk', 'rb'))
    val_pro_word_rcnn_char_rnn = pickle.load(open(Config.cache_dir + '/probe/word_rcnn_char_rnn/valid/2.pk', 'rb'))
    val_pro_word_rnn_char_rcnn = pickle.load(open(Config.cache_dir + '/probe/word_rnn_char_rcnn/valid/2.pk', 'rb'))
    val_pro_word_char_cgru = pickle.load(open(Config.cache_dir + '/probe/word_char_cgru/valid/1.pk', 'rb'))
    val_pro_word_rcnn_char_cgru = pickle.load(open(Config.cache_dir + '/probe/word_rcnn_char_cgru/valid/0.pk', 'rb'))
    val_pro_word_cgru_char_rnn = pickle.load(open(Config.cache_dir + '/probe/word_cgru_char_rnn/valid/0.pk', 'rb'))
    val_pro_word_cgru_char_rcnn = pickle.load(open(Config.cache_dir + '/probe/word_cgru_char_rcnn/valid/0.pk', 'rb'))
    val_pro_word_rnn_char_cgru = pickle.load(open(Config.cache_dir + '/probe/md/word_rnn_char_cgru/valid/2.pk', 'rb'))
    val_pro_word_rnn_char_cnn = pickle.load(open(Config.cache_dir + '/probe/md/word_rnn_char_cnn/valid/0.pk', 'rb'))

    val_pro = val_pro_cnn
    val_pro += val_pro_rnn
    val_pro += val_pro_rcnn
    val_pro += val_pro_deep_cnn
    val_pro += val_pro_word_rcnn_char_rnn
    val_pro += val_pro_word_rnn_char_rcnn
    val_pro += val_pro_word_char_cgru
    val_pro += val_pro_word_cgru_char_rcnn
    val_pro += val_pro_word_rcnn_char_cgru
    val_pro += val_pro_word_cgru_char_rnn
    val_pro += val_pro_word_rnn_char_cgru
    val_pro += val_pro_word_rnn_char_cnn

    val_pred = np.argmax(val_pro, axis=1)
    val_pred_label = to_label(val_pred)
    submit_result(val_ids, val_pred_label, Config.validation_submit_path)


def ensemble_final():
    final = pd.read_csv(Config.final_data_path, sep='\t')
    val_ids = final.id.values
    final_pro_deep_word_char_cnn = pickle.load(open(Config.cache_dir + '/final_probe_v2/deep_word_char_cnn/0.pk', 'rb'))
    final_pro = final_pro_deep_word_char_cnn
    final_pred = np.argmax(final_pro, axis=1)
    final_pred_label = to_label(final_pred)
    submit_result(val_ids, final_pred_label, Config.final_submit_path)


