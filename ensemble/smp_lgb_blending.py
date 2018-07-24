import pickle
import time
import xgboost as xgb
import numpy as np
import codecs
import gc
import os
import csv
from sklearn.cross_validation import cross_val_score
import pickle
import time
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

'''
:param

这个是模型融合的blending方法
'''

#tag dict done......
dict_tag  = {"自动摘要":0, "机器翻译":1, "机器作者":2, "人类作者":3}

dict_new= dict(zip(dict_tag.values(), dict_tag.keys()))



def myAcc(y_true,y_pred):
    #最大数的索引
    # y_pred = np.argmax(y_pred,axis=1)
    predict_num = np.mean(y_true == y_pred)
    return predict_num

def xgb_acc_score(preds,dtrain):
    y_true = dtrain.get_label()
    y_pred = np.argmax(preds,axis=1)
    return [('acc',np.mean(y_true == y_pred))]




# --------------------------------------------- train data ---------------------------------------------
'''
:param
遍历文件夹下面的所有文件
'''
print ('building train test data ...')
list_path = []
for i in os.listdir('./probe'):
    print (i)
    list_path.append(i)

# begin model vector
train_vec_path_try = './probe/' + 'cnn' + '/test/' + str(0) + '.pk'
X2 = pickle.load(open(train_vec_path_try, 'rb'))

X2 = np.zeros((X2.shape[0],1))

for path in list_path:
    for num in range(0,3):
        train_vec_path = './probe/' + path + '/test/' + str(num) + '.pk'
        X_train = pickle.load(open(train_vec_path, 'rb'))
        # X_train = preprocessing.normalize(X_train)
        X2 = np.hstack((X2,X_train))

X2 = np.delete(X2, 0, axis=1)


# --------------------------------------------- test data ---------------------------------------------

# begin model vector
valid_vec_path_try = './probe/' + 'cnn' + '/valid/' + str(0) + '.pk'
X_valid = pickle.load(open(valid_vec_path_try, 'rb'))

X_valid = np.zeros((X_valid.shape[0],1))

for path in list_path:
    for num in range(0,3):
        valid_vec_path = './probe/' + path + '/valid/' + str(num) + '.pk'
        X_valid_file = pickle.load(open(valid_vec_path, 'rb'))
        # X_valid_file = preprocessing.normalize(X_valid_file)
        X_valid = np.hstack((X_valid,X_valid_file))

X_valid = np.delete(X_valid, 0, axis=1)


# --------------------------------------------tag process----------------------------------------

list_tag = []
file_path = '/media/iiip/文档/smp/new_data/train_data/test.csv'
val = pd.read_csv(file_path, sep='\t')
y = val.label
for i in y:
    list_tag.append(dict_tag[i])

# ---------------------------------------------mean array ----------------------------------------

def mean_arr(list_vec):
    list_tag0 = list_vec[:, ::4].mean(1)
    list_tag1 = list_vec[:, 1::4].mean(1)
    list_tag2 = list_vec[:, 2::4].mean(1)
    list_tag3 = list_vec[:, 3::4].mean(1)


    list_tag0 = np.stack((list_tag0,list_tag1),axis=1)
    list_tag1 = np.stack((list_tag2, list_tag3), axis=1)

    list_tag_con = np.hstack((list_tag0, list_tag1))

    del list_tag0,list_tag1,list_tag2,list_tag3
    gc.collect()

    return list_tag_con


def deal_training_arr(X):

    '''
    :param X:

    这里取每个维度对应的长度来进行平均化

      88vec : 原始的CNN  + HAN
    1 layer : 新的300维度的CNN
    2 layer : 新的300维度的CRNN
    3 layer : 新的300维度的HAN
    4 layer : 新的300维度的RNN

    可根据输入的维度不同来改下列的范围
    :return:
    '''

    list_cnn_vec = X[:,:4]
    list_crnn_vec = X[:,4:8]
    list_han_vec = X[:,8:12]
    list_rnn_vec = X[:,12:]
    # list_han_prob_vec = X[:,120:]
    # list_crnn_prob_vec = X[:,120:120]
    # list_cnn_prob_vec = X[:, 80:120]

    # split 4
    list_vec_concat_cnn = mean_arr(list_cnn_vec)
    list_vec_concat_crnn = mean_arr(list_crnn_vec)
    list_vec_concat_han = mean_arr(list_han_vec)
    list_vec_concat_rnn = mean_arr(list_rnn_vec)


    # cnn-han-8
    list_concat = np.hstack((list_vec_concat_cnn,list_vec_concat_crnn))
    list_concat = np.hstack((list_concat, list_vec_concat_han))
    list_concat = np.hstack((list_concat,list_vec_concat_rnn))

    X = np.hstack((X,list_concat))


    del list_cnn_vec,list_han_vec,list_vec_concat_cnn,list_vec_concat_han,list_concat
    gc.collect()

    print ('total concat done----------------------------------')
    print (X.shape)

    return X

# ----------------------------------------------分布型向量 --------------------------------------------
print ('building disperse vec ...')

csv_file = csv.reader(open('./features/test_result.csv','r'))

X_disperse_vec_train= []

for stu in csv_file:
    if('ppl_range' in stu):
        pass
    else:
        X_disperse_vec_train.append( stu[1:])


csv_file = csv.reader(open('./features/validation_result.csv','r'))

X_disperse_vec_valid= []

for stu in csv_file:
    if('ppl_range' in stu):
        pass
    else:
        X_disperse_vec_valid.append( stu[1:])

X_disperse_vec_train = np.array(X_disperse_vec_train)
X_disperse_vec_valid = np.array(X_disperse_vec_valid)

# ---------------------------------------------split data ----------------------------------------


X = np.hstack((X2,X_disperse_vec_train))
X_valid = np.hstack((X_valid,X_disperse_vec_valid))

del X2
gc.collect()

print ('train shape :-------------------------')
print (X.shape)
print ()
print ('test shape :-------------------------')
print (X_valid.shape)

from sklearn.cross_validation import train_test_split
#x为数据集的feature熟悉，y为label.
x_d1, x_d2, y_d1, y_d2 = train_test_split(X, list_tag, test_size = 0.3,random_state=2018)

del X,list_tag,y
gc.collect()

# ----------------------------------------------blending-------------------------------------------
'''模型融合中使用到的各个单模型'''

clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        lgb.LGBMClassifier(num_class=4,max_depth=-1,n_estimators=20,objective='multiclass',learning_rate=0.01,num_leaves=65,
    #new bagging
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    #new bagging
                           n_jobs=-1,
),

lgb.LGBMClassifier(num_class=4,max_depth=-1,n_estimators=20,objective='multiclass',learning_rate=0.01,num_leaves=65,
    #new bagging
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=10,
    #new bagging
                           n_jobs=-1,
),
        ]

dataset_d1 = np.zeros((x_d2.shape[0], len(clfs)))
dataset_d2 = np.zeros((X_valid.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    print(j, clf)
    '''使用第1个部分作为预测，第2部分来训练模型，获得其预测的输出作为第2部分的新特征。'''
    # X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    clf.fit(x_d1,y_d1)
    y_submission = clf.predict(x_d2)
    print (y_submission)
    dataset_d1[:, j] = y_submission
    '''对于测试集，直接用这k个模型的预测值作为新的特征。'''
    dataset_d2[:, j] = clf.predict(X_valid)


# ----------------------------------------------lgb -----------------------------------------------
def LGB_test():
    print("LGB test")
    clf = lgb.LGBMClassifier(
        num_class=4,
        max_depth=-1,
        n_estimators=5000,
        objective='multiclass',
        learning_rate=0.01,
        num_leaves=65,

        # #new bagging
        # feature_fraction=0.9,
        # bagging_fraction=0.8,
        # bagging_freq=5,
        # #new bagging

        n_jobs=-1,
    )

    print (dataset_d1)
    print (dataset_d2)
    time.sleep(20)
    print ()
    clf.fit(dataset_d1, y_d2, eval_set=[(dataset_d1, y_d2)], eval_metric='logloss', early_stopping_rounds=1000)
    y_pred_te = clf.predict(dataset_d2,num_iteration=clf.best_iteration_)

    print(y_pred_te)


    return y_pred_te
    # print("多分类 + lightgbm　准确率平均值为: ")
    # # #获取准确率
    # print(myAcc(y_test, y_pred_te))

# ---------------------------------------------split data ----------------------------------------


def write_proj(y_pred_te):

    f = codecs.open('temp_mean.txt','w')
    for i in y_pred_te:
        f.write(dict_new[i])
        f.write("\n")

    print ('ok--------------------------------------------------------')

    list_total_user = []
    f = codecs.open('validation_id.txt')
    for i in f.readlines():
        i = i.replace('\n', '').strip()
        list_total_user.append(i)

    with open("_temp_132.csv", 'w') as file:
        for i in range(len(y_pred_te)):
            s = str(list_total_user[i]) + "," + str(dict_new[y_pred_te[i]])
            # print(s)
            file.write(s)
            file.write("\n")
    print("finish to write result")


if __name__ == '__main__':
    preds = LGB_test()
    write_proj(preds)


