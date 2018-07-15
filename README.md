# SMP 2018
distinguish human writing or robot writing from articles

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
  - `my_utils/data_preprocess`: create data sequence and batch for the input of deep learning model
  - `my_utils/w2v_process`: get the vocabs and pre-trained embeddings for words and chars
  - `my_utils/metrics`: calcuate the precision, recall and F1 scores for each categories of authors

## 3.models
