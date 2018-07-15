# coding=utf-8

import sys
from collections import defaultdict

sys.path.append("..")

from init.config import Config
from data import *

import os
import multiprocessing
import gensim
import re
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from joblib import Parallel, delayed
import pickle
import numpy as np
import word2vec

# overwrite = True
vec_dim = 256


def create_w2vc(overwrite=True):
    if overwrite:
        if os.path.exists(Config.cache_dir + '/w2v_dataframe.csv'):
            os.remove(Config.cache_dir + '/w2v_dataframe.csv')
        if os.path.exists(Config.cache_dir + '/w2v_content_word.txt'):
            os.remove(Config.cache_dir + '/w2v_content_word.txt')
            # f = open(Config.cache_dir + '/w2v_content_word.txt', 'w')
            # f.close()
        if os.path.exists(Config.cache_dir + '/w2v_content_char.txt'):
            os.remove(Config.cache_dir + '/w2v_content_char.txt')
            # f = open(Config.cache_dir + '/w2v_content_char.txt', 'w')
            # f.close()

        train_data = get_train_all_data()
        vali_data = get_validation_data()

        train_content = train_data["content"]
        vali_content = vali_data["content"]

        contents = train_content + vali_content
        print("len of train contents", len(train_content))
        print ("len of vali contents", len(vali_content))
        print("len of contents:", len(contents))

        def applyParallel(contents, func, n_thread):
            with Parallel(n_jobs=n_thread) as parallel:
                parallel(delayed(func)(c) for c in contents)

        def word_content(content):
            with open(Config.cache_dir + "/w2v_content_word.txt", "a+") as f:
                f.writelines(content.lower())
                f.writelines('\n')

        def char_content(content):
            with open(Config.cache_dir + "/w2v_content_char.txt", "a+") as f:
                content = content.lower().replace(" ", "")
                # print(content)
                f.writelines(" ".join(content))
                f.writelines("\n")

        applyParallel(train_content, word_content, 25)
        applyParallel(train_content, char_content, 25)
        applyParallel(vali_content, word_content, 25)
        applyParallel(vali_content, char_content, 25)


    # word vector train
    model = gensim.models.Word2Vec(
        LineSentence(Config.cache_dir + "/w2v_content_word.txt"),
        size=vec_dim,
        window=5,
        min_count=1,
        workers=multiprocessing.cpu_count()
    )
    model.save(Config.cache_dir + "/content_w2v_word.model")

    # char vector train
    model = gensim.models.Word2Vec(
        LineSentence(Config.cache_dir + '/w2v_content_char.txt'),
        size=vec_dim,
        window=5,
        min_count=1,
        workers=multiprocessing.cpu_count()
    )
    model.save(Config.cache_dir + "/content_w2v_content_char.model")

def convert_num(word):
    pattern = re.compile('[0-9]+')
    match = pattern.findall(word)
    if match:
        return True
    else:
        return False


def create_word_vocab(overwriter=False):
    word_freq = defaultdict(int)

    train_data = get_train_all_data()
    vali_data = get_validation_data()
    train_content = train_data["content"]
    vali_content = vali_data["content"]

    for line in train_content:
        line = line.lower().strip()
        words = line.split(" ")
        for word in words:
            if " " == word or "" == word:
                continue
            word_freq[word] += 1

    for line in vali_content:
        line = line.lower().strip()
        words = line.split(" ")
        for word in words:
            if " " == word or "" == word:
                continue
            word_freq[word] += 1
    vocab = {}
    i = 1
    min_freq = 1
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = i
            i += 1
    vocab['NUM'] = i
    vocab['UNK'] = i+1
    print("size of vocab:", len(vocab))

    if overwriter:
        vocab_file = Config.cache_dir + '/word_vocab.pk'
        with open(vocab_file, 'wb') as f:
            pickle.dump(vocab, f)
        print("finish to create vocab")


def create_char_vocab(overwriter=False):
    char_freq = defaultdict(int)

    train_data = get_train_all_data()
    vali_data = get_validation_data()
    train_content = train_data["content"]
    vali_content = vali_data["content"]

    for line in train_content:
        line = line.lower().strip()
        line = line.replace(" ", "")
        chars_line = " ".join(line)
        chars = chars_line.split(" ")
        for char in chars:
            if " " == char or "" == char:
                continue
            char_freq[char] += 1

    for line in vali_content:
        line = line.lower().strip()
        line = line.replace(" ", "")
        chars_line = " ".join(line)
        chars = chars_line.split(" ")
        for char in chars:
            if " " == char or "" == char:
                continue
            char_freq[char] += 1
    vocab = {}
    i = 1
    min_freq = 1
    for char, freq in char_freq.items():
        if freq >= min_freq:
            vocab[char] = i
            i += 1
    vocab['NUM'] = i
    vocab['UNK'] = i+1
    print(vocab)
    print("size of vocab:", len(vocab))

    if overwriter:
        vocab_file = Config.cache_dir + '/char_vocab.pk'
        with open(vocab_file, 'wb') as f:
            pickle.dump(vocab, f)
        print("finish to create vocab")


def create_word_emb(use_opened=True, overwriter=False):

    vocab = pickle.load(open(Config.cache_dir + '/word_vocab.pk', 'rb'))
    print(len(vocab))

    if use_opened:
        word_emb = [np.random.uniform(0, 0, 200) for j in range(len(vocab)+1)]
        model = word2vec.load(Config.open_w2v_path)
    else:
        word_emb = [np.random.uniform(0, 0, 256) for j in range(len(vocab)+1)]
        model = gensim.models.Word2Vec.load(Config.cache_dir + "/content_w2v_word.model")
    num = 0
    # print (len(vocab))
    for word in vocab:
        index = vocab[word]
        # print(index, word)
        if word in model:
            word_emb[index] = np.array(model[word])
            num += 1
        else:
            word_emb[index] = np.random.uniform(-0.5, 0.5, 200)
    word_emb = np.array(word_emb)
    print("word number: ", num)
    print("vocab size:", len(vocab))
    print("shape of word_emb", np.shape(word_emb))
    if overwriter:
        with open(Config.word_embed_path, 'wb') as f:
            pickle.dump(word_emb, f)
            print("size of embedding_matrix: ", len(word_emb))
            print("word_embedding finish")


def create_char_emb(overwriter=False):

    vocab = pickle.load(open(Config.char_vocab_path, 'rb'))
    char_emb = [np.random.uniform(0, 0, 256) for j in range(len(vocab)+1)]
    model = gensim.models.Word2Vec.load(Config.cache_dir + "/content_w2v_word.model")
    num = 0
    for char in vocab:
        index = vocab[char]
        if char in model:
            char_emb[index] = np.array(model[char])
            num += 1
        else:
            char_emb[index] = np.random.uniform(-0.5, 0.5, 256)
    char_emb = np.array(char_emb)
    print("char number: ", num)
    print("vocab size:", len(vocab))
    print("shape of char_emb", np.shape(char_emb))
    if overwriter:
        with open(Config.char_embed_path, 'wb') as f:
            pickle.dump(char_emb, f)
            print("size of embedding_matrix: ", len(char_emb))
            print("char_embedding finish")



# create_word_vocab(overwriter=True)
# create_char_vocab(overwriter=True)
# create_word_emb(use_opened=True, overwriter=True)
# create_char_emb(overwriter=True)