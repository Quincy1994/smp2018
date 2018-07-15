# coding=utf-8
import codecs
import re
import pandas as pd
from init.config import Config
from sklearn.model_selection import train_test_split
from pyltp import SentenceSplitter

cfg = Config()
label_dict = {"自动摘要": 0, "机器翻译": 1, "机器作者": 2, "人类作者": 3}

def create_training_data(save_all_samples=False, save_split_sample=False):
    data = []
    data_error_rows = []
    with open(cfg.train_org_path) as input_file:
        lines = input_file.readlines()
        for line in lines:
            line = line.strip()
            label = re.search('label:(.*?), contence', line).group(1)
            content = re.search('contence:(.*?), id', line).group(1)
            id = re.search('id:(.*?)}', line).group(1)
            if label is None or content is None or id is None:
                row = {}
                row["id"] = id
                row["label"] = label
                row["content"] = content
                data_error_rows.append(row)
            else:
                row = {}
                row["id"] = id
                row["label"] = label
                row["content"] = content
                data.append(row)
    print("total samples number:", len(data))
    print("wrong samples number:", len(data_error_rows))
    data = pd.DataFrame(data)
    data = data[["id", "content", "label"]]
    if save_all_samples:
        data.fillna("", inplace=True)
        data.to_csv(cfg.train_all_data_path, index=False, sep='\t')
    train, val = train_test_split(data, test_size=0.1, shuffle=True, random_state=1)
    print("train samples number:", len(train))
    print("vali samples number:", len(val))
    if save_split_sample:
        train.to_csv(cfg.train_data_path, index=False, sep='\t')
        val.to_csv(cfg.test_data_path, index=False, sep='\t')
    return train, val

def create_validation_data(save_samples=False):
    data = []
    data_error_rows = []
    with open(cfg.vali_org_path) as input_file:
        lines = input_file.readlines()
    with open(cfg.vali_id_path) as id_file:
        ids = id_file.readlines()

    for i in range(len(lines)):
        content = lines[i].strip()
        id = ids[i].strip()
        if content is None or id is None:
            row = {}
            row["id"] = id
            row["content"] = content
            data_error_rows.append(row)
        else:
            row = {}
            row["id"] = id
            row["content"] = content
            data.append(row)
    print("total samples number:", len(data))
    print("wrong samples number:", len(data_error_rows))
    data = pd.DataFrame(data)
    data = data[["id", "content"]]
    if save_samples:
        data.fillna("", inplace=True)
        data.to_csv(cfg.vali_data_path, index=False, sep='\t')


def get_train_all_data():
    train_all_data = pd.read_csv(Config.train_all_data_path, sep='\t')
    return train_all_data


def get_train_split_data():
    train = pd.read_csv(Config.train_data_path, sep='\t')
    test = pd.read_csv(Config.test_data_path, sep='\t')
    return train, test

def get_validation_data():
    vali = pd.read_csv(Config.vali_data_path, sep='\t')
    return vali


def save_dict(dict, save_path):
    f = open(save_path, "w")
    for key in dict:
        f.write(str(key)+","+ str(dict[key]) +'\n')
    f.close()


def count_sentence_num_length():
    train_data = get_train_all_data()
    vali_data = get_validation_data()
    train_content = train_data["content"]
    train_labels = train_data["label"]
    vali_content = vali_data["content"]


    train_num_dict = {}
    vali_num_dict = {}
    auto_abs_train_num_dict = {}
    mach_trans_train_num_dict = {}
    mach_auth_train_num_dict = {}
    human_auth_train_num_dict = {}
    for i in range(len(train_content)):
        content = train_content[i].lower().strip()
        sentences = SentenceSplitter.split(content)
        sent_num = len(sentences)
        if sent_num in train_num_dict:
            train_num_dict[sent_num] +=1
        else:
            train_num_dict[sent_num] = 1
        if train_labels[i] == '自动摘要':
            if sent_num in auto_abs_train_num_dict:
                auto_abs_train_num_dict[sent_num] += 1
            else:
                auto_abs_train_num_dict[sent_num] = 1
        elif train_labels[i] == '机器翻译':
            if sent_num in mach_trans_train_num_dict:
                mach_trans_train_num_dict[sent_num] += 1
            else:
                mach_trans_train_num_dict[sent_num] =1
        elif train_labels[i] == '机器作者':
            if sent_num in mach_auth_train_num_dict:
                mach_auth_train_num_dict[sent_num] += 1
            else:
                mach_auth_train_num_dict[sent_num] = 1
        elif train_labels[i] == '人类作者':
            if sent_num in human_auth_train_num_dict:
                human_auth_train_num_dict[sent_num] += 1
            else:
                human_auth_train_num_dict[sent_num] = 1
        else:
            print("wrong", train_labels[i])
    save_dict(train_num_dict, Config.cache_dir + "/train_sen_num.csv")
    save_dict(auto_abs_train_num_dict, Config.cache_dir + "/auto_abs_sen_num.csv")
    save_dict(mach_trans_train_num_dict, Config.cache_dir + "/mach_trans_sen_num.csv")
    save_dict(mach_auth_train_num_dict, Config.cache_dir + "/mach_auth_sen_num.csv")
    save_dict(human_auth_train_num_dict, Config.cache_dir + "/human_auth_sen_num.csv")

    for content in vali_content:
        content = content.lower().strip()
        sentences = SentenceSplitter.split(content)
        sent_num = len(sentences)
        if sent_num not in vali_num_dict:
            vali_num_dict[sent_num] = 1
        else:
            vali_num_dict[sent_num] += 1
    save_dict(vali_num_dict, Config.cache_dir + "/vali_sen_num.csv")

def count_word_sentence_length():
    train_data = get_train_all_data()
    vali_data = get_validation_data()
    train_content = train_data["content"]
    train_labels = train_data["label"]
    vali_content = vali_data["content"]

    train_sent_len_dict = {}
    vali_sent_len_dict = {}
    auto_abs_train_sent_len_dict = {}
    mach_trans_train_sent_len_dict = {}
    mach_auth_train_sent_len_dict = {}
    human_auth_train_sent_len_dict = {}
    for i in range(len(train_content)):
        content = train_content[i].lower().strip()
        sentences = SentenceSplitter.split(content)
        for sent in sentences:
            sent = sent.strip()
            sent_len = len(sent.split(" "))
            print(sent)
            print(sent_len)
            if sent_len not in train_sent_len_dict:
                train_sent_len_dict[sent_len] = 1
            else:
                train_sent_len_dict[sent_len] += 1
            if train_labels[i] == '自动摘要':
                if sent_len not in auto_abs_train_sent_len_dict:
                    auto_abs_train_sent_len_dict[sent_len] = 1
                else:
                    auto_abs_train_sent_len_dict[sent_len] += 1
            elif train_labels[i] == '机器翻译':
                if sent_len not in mach_trans_train_sent_len_dict:
                    mach_trans_train_sent_len_dict[sent_len] = 1
                else:
                    mach_trans_train_sent_len_dict[sent_len] += 1
            elif train_labels[i] == '机器作者':
                if sent_len not in mach_auth_train_sent_len_dict:
                    mach_auth_train_sent_len_dict[sent_len] = 1
                else:
                    mach_auth_train_sent_len_dict[sent_len] += 1
            elif train_labels[i] == '人类作者':
                if sent_len not in human_auth_train_sent_len_dict:
                    human_auth_train_sent_len_dict[sent_len] = 1
                else:
                    human_auth_train_sent_len_dict[sent_len] += 1
            else:
                print("wrong", train_labels[i])
    save_dict(train_sent_len_dict, Config.cache_dir + "/train_word_sen_len.csv")
    save_dict(auto_abs_train_sent_len_dict, Config.cache_dir + "/auto_abs_word_sen_len.csv")
    save_dict(mach_trans_train_sent_len_dict, Config.cache_dir + "/mach_trans_word_sen_len.csv")
    save_dict(mach_auth_train_sent_len_dict, Config.cache_dir + "/mach_auth_word_sen_len.csv")
    save_dict(human_auth_train_sent_len_dict, Config.cache_dir + "/human_auth_word_sen_len.csv")

    for content in vali_content:
        content = content.lower().strip()
        sentences = SentenceSplitter.split(content)
        for sent in sentences:
            sent = sent.strip()
            sent_len = len(sent.split(" "))
            if sent_len not in vali_sent_len_dict:
                vali_sent_len_dict[sent_len] = 1
            else:
                vali_sent_len_dict[sent_len] += 1
    save_dict(vali_sent_len_dict, Config.cache_dir + "/vali_word_sent_len.csv")

def count_char_sentence_length():
    train_data = get_train_all_data()
    vali_data = get_validation_data()
    train_content = train_data["content"]
    train_labels = train_data["label"]
    vali_content = vali_data["content"]

    train_sent_len_dict = {}
    vali_sent_len_dict = {}
    auto_abs_train_sent_len_dict = {}
    mach_trans_train_sent_len_dict = {}
    mach_auth_train_sent_len_dict = {}
    human_auth_train_sent_len_dict = {}
    for i in range(len(train_content)):
        content = train_content[i].lower().strip()
        sentences = SentenceSplitter.split(content)
        for sent in sentences:
            sent = sent.strip()
            line = sent.replace(" ", "")
            chars_line = " ".join(line)
            chars = chars_line.split(" ")
            chars_len = len(chars)
            # print(chars_line)
            # print(chars_len)
            if chars_len not in train_sent_len_dict:
                train_sent_len_dict[chars_len] = 1
            else:
                train_sent_len_dict[chars_len] += 1
            if train_labels[i] == '自动摘要':
                if chars_len not in auto_abs_train_sent_len_dict:
                    auto_abs_train_sent_len_dict[chars_len] = 1
                else:
                    auto_abs_train_sent_len_dict[chars_len] += 1
            elif train_labels[i] == '机器翻译':
                if chars_len not in mach_trans_train_sent_len_dict:
                    mach_trans_train_sent_len_dict[chars_len] = 1
                else:
                    mach_trans_train_sent_len_dict[chars_len] += 1
            elif train_labels[i] == '机器作者':
                if chars_len not in mach_auth_train_sent_len_dict:
                    mach_auth_train_sent_len_dict[chars_len] = 1
                else:
                    mach_auth_train_sent_len_dict[chars_len] += 1
            elif train_labels[i] == '人类作者':
                if chars_len not in human_auth_train_sent_len_dict:
                    human_auth_train_sent_len_dict[chars_len] = 1
                else:
                    human_auth_train_sent_len_dict[chars_len] += 1
            else:
                print("wrong", train_labels[i])
    save_dict(train_sent_len_dict, Config.cache_dir + "/train_char_sen_len.csv")
    save_dict(auto_abs_train_sent_len_dict, Config.cache_dir + "/auto_abs_char_sen_len.csv")
    save_dict(mach_trans_train_sent_len_dict, Config.cache_dir + "/mach_trans_char_sen_len.csv")
    save_dict(mach_auth_train_sent_len_dict, Config.cache_dir + "/mach_auth_char_sen_len.csv")
    save_dict(human_auth_train_sent_len_dict, Config.cache_dir + "/human_auth_char_sen_len.csv")

    for content in vali_content:
        content = content.lower().strip()
        sentences = SentenceSplitter.split(content)
        for sent in sentences:
            line = sent.replace(" ", "")
            chars_line = " ".join(line)
            chars = chars_line.split(" ")
            chars_len = len(chars)
            if chars_len not in vali_sent_len_dict:
                vali_sent_len_dict[chars_len] = 1
            else:
                vali_sent_len_dict[chars_len] += 1
    save_dict(vali_sent_len_dict, Config.cache_dir + "/vali_char_sent_len.csv")





# count_sentence_num_length()
# count_word_sentence_length()
# count_char_sentence_length()

# create_training_data(save_all_samples=True, save_split_sample=True)
# create_validation_data(save_samples=False)