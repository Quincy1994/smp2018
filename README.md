# SMP 2018 (1st prize)
This contest is to distinguish human writing or robot writing from articles, and we won the champion out of 240 teams.

# Task description
Given an article, we need to create algorithms that judge types of authors (automatic summary, machine translation, robot writer or human writer). 
More details see [SMP EUPT 2018](https://www.biendata.com/competition/smpeupt2018/)

## 1.Set up
* tensorflow >= 1.4.0
* keras >= 1.2.0
* gensim
* scikit-learn
you may need **keras.utils.vis_utils** for model visualization 

## 2.Data Preprocessing
- `my_utils/`: for data preprocessing
  - `my_utils/data`: convert origin data to csv file
  - `my_utils/data_preprocess`: create data sequences and batches for the input of deep learning models
  - `my_utils/w2v_process`: get the vocabs and pre-trained embeddings for words and chars
  - `my_utils/metrics`: calcuate the precision, recall and F1 scores for each categories of authors

## 3.Models
There are total 12 models that combine word representations and character representations.
The best model `word rcnn char cgru` we devised is spired by two papers:
* [A Hybrid Framework for Text Modeling with Convolutional RNN](http://xueshu.baidu.com/s?wd=paperuri%3A%288fa9aee951dcbd75f9259bc0f6bee7d6%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fdl.acm.org%2Fcitation.cfm%3Fid%3D3098140&ie=utf-8&sc_us=15226213875739465170)
* [A C-LSTM Neural Network for Text Classfication](http://xueshu.baidu.com/s?wd=paperuri%3A%28e3c8a546d60164116642a41cca6f2ad8%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fpdf%2F1511.08630&ie=utf-8&sc_us=5294540248844921011)

Here is the scores of different models:

 |model |off-line |on-line |
 :---: |:---: |:---:
word_char_cnn | 0.9888 | 0.9849
word_char_rnn | 0.9894 | 0.9863
deep_word_char_cnn | 0.9887 | 0.9828
word_rcnn_char_rnn | 0.9899 | 0.9879
word_rnn_char_rcnn | 0.9902 | 0.9872
word_char_cgru | 0.9896 | 0.9861
word_cgru_char_rcnn | 0.9904 | untested
word_rcnn_char_cgru | 0.9910 | 0.9882
word_cgru_char_rnn | 0.9887 | untested
word_rnn_char_cgru | 0.9899 | untested
word_rnn_char_cnn | 0.9897 | 0.9862
word_char_rcnn | 0.9894 | 0.9984

* Note that rcnn comes from `A Hybrid Framework for Text Modeling with Convolutional RNN` while cgru comes from `A C-LSTM Neural Network for Text Classfication`

The source codes derives from https://github.com/fuliucansheng/360
We use `model` to create the architectures of models, and use `train` to train them

## 4.ensemble
We use a simple but efficient voting mechanism for ensembling, and we can find it in `evaluate/predict`

## 5.Main files
- `my_utils/`: for data preprocessing
  - `my_utils/data`: convert origin data to csv file
  - `my_utils/data_preprocess`: create data sequences and batches for the input of deep learning models
  - `my_utils/w2v_process`: get the vocabs and pre-trained embeddings for words and chars
  - `my_utils/metrics`: calcuate the precision, recall and F1 scores for each categories of authors
- `models/`: for creating deep learning models
  - `deepzoo`: for keeping all models
- `init/config.py`: for saving the path of models, data and so on
- `train`: for training models
- `figure`: for saving the visualization of models

# Acknowledgment
Thanks for all the efforts of my teammates in `GDUFS-iiip` 
We hope that more people will join in our labs: `Data Mining Lab in GDUFS(广外数据挖掘实验室）`
