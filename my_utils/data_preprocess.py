# coding=utf-8
from init.config import Config
import pickle
import re
import numpy as np
from keras.preprocessing import sequence
from my_utils.data import *
from pyltp import SentenceSplitter

word_vocab = pickle.load(open(Config.word_vocab_v2_path, 'rb'))
char_vocab = pickle.load(open(Config.char_vocab_v2_path, 'rb'))

def convert_num(word):
    pattern = re.compile('[0-9]+')
    match = pattern.findall(word)
    if match:
        return True
    else:
        return False

def get_word_seq(contents, word_maxlen=Config.word_seq_maxlen, mode="post", keep=False, verbose=False):
    unknow_index =len(word_vocab)
    word_r = []
    for content in contents:
        word_c = []
        content = content.lower().strip()
        words = content.split(" ")
        for word in words:
            if convert_num(word):
                word = 'NUM'
            if word in word_vocab:
                index = word_vocab[word]
            else:
                index = unknow_index
            word_c.append(index)
        word_c = np.array(word_c)
        word_r.append(word_c)
    word_seq = sequence.pad_sequences(word_r, maxlen=word_maxlen, padding=mode, truncating=mode, value=0)
    # print (word_seq)
    return word_seq


def get_char_seq(contents, char_maxlen=Config.char_seq_maxlen, mode='post', keep=False, verbost=False):
    unknow_index = len(char_vocab)
    char_r = []
    for content in contents:
        char_c = []
        content = content.lower().strip()
        content = content.replace(" ", "")
        chars_line = " ".join(content)
        chars = chars_line.split(" ")
        for char in chars:
            if convert_num(char):
                char = 'NUM'
            if char in char_vocab:
                index = char_vocab[char]
            else:
                index = unknow_index
            char_c.append(index)
        char_c = np.array(char_c)
        char_r.append(char_c)
    char_seq = sequence.pad_sequences(char_r, maxlen=char_maxlen, padding=mode, truncating=mode, value=0)
    return char_seq


def to_categorical(labels):
    label_dict = {"自动摘要": 0, "机器翻译": 1, "机器作者": 2, "人类作者": 3}
    y = []
    for label in labels:
        y_line = [0, 0, 0, 0]
        y_line[label_dict[label]] = 1
        y.append(y_line)
    y = np.array(y)
    return y


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)* batch_size)) for i in range(0, nb_batch)]

def batch_generator(contents, labels, batch_size=128, shuffle=True, keep=False, preprocessfunc=None):

    assert preprocessfunc != None
    sample_size = contents.shape[0]
    index_array = np.arange(sample_size)

    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start: batch_end]
            batch_contents = contents[batch_ids]
            batch_contents = preprocessfunc(batch_contents, keep=keep)
            batch_labels = to_categorical(labels[batch_ids])
            yield (batch_contents, batch_labels)


def word_cnn_preprocess(contents, word_maxlen=Config.word_seq_maxlen, keep=False):
    word_seq = get_word_seq(contents, word_maxlen=word_maxlen, keep=keep)
    return word_seq


def word_han_preprocess(contents, sentence_num=Config.sentence_num, sentence_length=Config.sentence_word_length, keep=False):
    contents_seq = np.zeros(shape=(len(contents), sentence_num, sentence_length))
    for index, content in enumerate(contents):
        if index >= len(contents): break
        # print (index)
        sentences = SentenceSplitter.split(content)
        word_seq = get_word_seq(sentences, word_maxlen=sentence_length)
        word_seq = word_seq[:sentence_num]
        contents_seq[index][:len(word_seq)] = word_seq
    return contents_seq


def word_cnn_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label,batch_size=batch_size, keep=keep, preprocessfunc=word_cnn_preprocess)


def word_han_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=word_han_preprocess)


def char_cnn_preprocess(contents, maxlen=Config.char_seq_maxlen, keep=False):
    char_seq = get_char_seq(contents, char_maxlen=maxlen, keep=keep)
    return char_seq


def char_han_preprocess(contents, sentence_num=Config.sentence_num, sentence_length=Config.sentence_char_length, keep=False):
    content_seq = np.zeros(shape=(len(contents), sentence_num, sentence_length))
    for index, content in enumerate(contents):
        if index >= (len(contents)): break
        sentences = SentenceSplitter.split(content)
        char_seq = get_char_seq(sentences,char_maxlen=sentence_length)
        char_seq = char_seq[:sentence_num]
        content_seq[index][:len(char_seq)] = char_seq
    return content_seq


def char_cnn_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=char_cnn_preprocess)


def char_han_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=char_han_preprocess)

def word_char_cnn_preprocess(contents , word_maxlen=Config.word_seq_maxlen, char_maxlen=Config.char_seq_maxlen, keep=False):
    word_seq = get_word_seq(contents, word_maxlen=word_maxlen, keep=keep)
    char_seq = get_char_seq(contents, char_maxlen=char_maxlen, keep=keep)
    return [word_seq, char_seq]

def word_char_han_preprocess(contents, sentence_num=Config.sentence_num, sentence_word_length=Config.sentence_word_length, sentence_char_length=Config.sentence_char_length, keep=False):
    contents_word_seq = np.zeros(shape=(len(contents), sentence_num, sentence_word_length))
    contents_char_seq = np.zeros(shape=(len(contents), sentence_num, sentence_char_length))
    for index, content in enumerate(contents):
        if index >= (len(contents)): break
        sentences = SentenceSplitter.split(content)
        word_seq = get_word_seq(sentences, word_maxlen=sentence_word_length)
        word_seq = word_seq[:sentence_num]
        char_seq = get_char_seq(sentences, char_maxlen=sentence_char_length)
        char_seq = char_seq[:sentence_num]
        contents_word_seq[index][:len(word_seq)] = word_seq
        contents_char_seq[index][:len(char_seq)] = char_seq
    return [contents_word_seq, contents_char_seq]

def word_char_cnn_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=word_char_cnn_preprocess)

def word_char_han_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=word_char_han_preprocess)









# train_data = get_train_all_data()
# vali_data = get_validation_data()
# train_content = train_data["content"]
# vali_content = vali_data["content"]
#
# get_word_seq(train_content)
# get_word_seq(vali_content)
# get_char_seq(train_content)
# get_char_seq(vali_content)

