#  coding=utf-8

import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer
from keras.backend.tensorflow_backend import set_session
# from recurrentshop import *

from model.Attention import *
from model.Capsule import *


def convs_block(data, convs = [3,4,5], f = 256, name = "conv_feat"):
    pools = []
    for c in convs:
        conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=f, kernel_size=c, padding="valid")(data)))
        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools, name=name)

# def inception_convs(data, convs=[3,4,5], f = )



# 简单的cnn
def get_textcnn(seq_length, embed_weight):
    content = Input(shape=(seq_length,), dtype="int32")
    embedding = Embedding(
        name="embedding",
        input_dim=embed_weight.shape[0],
        weights=[embed_weight],
        output_dim=embed_weight.shape[1],
        trainable=False)
    trans_content = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(content)))))
    feat = convs_block(trans_content)
    dropfeat = Dropout(0.2)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(4, activation="softmax")(fc)
    model = Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


def get_hcnn(sent_num, sent_length, embed_weight, mask_zero=False):
    sentence_input = Input(shape=(sent_length, ), dtype="int32")
    embedding = Embedding(
        input_dim = embed_weight.shape[0],
        weights = [embed_weight],
        output_dim=embed_weight.shape[1],
        mask_zero = mask_zero,
        trainable = False
    )
    sent_embed = embedding(sentence_input)
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_embed)
    word_attention = Attention(sent_length)(word_bigru)
    sent_encode = Model(sentence_input, word_attention)

    review_input = Input(shape=(sent_num, sent_length), dtype="int32")
    review_encode = TimeDistributed(sent_encode)(review_input)
    feat = convs_block(review_encode)
    dropfeat = Dropout(0.2)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(2, activation="softmax")(fc)
    model = Model(review_input, output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


def get_han2(sent_num, sent_length, embed_weight, mask_zero=False):
    input = Input(shape=(sent_num, sent_length,), dtype="int32")
    embedding = Embedding(
        name= "embeeding",
        input_dim = embed_weight.shape[0],
        weights = [embed_weight],
        output_dim = embed_weight.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )
    sent_embed = embedding(input)
    # print(np.shape(sent_embed))
    sent_embed = Reshape((1, sent_length, embed_weight.shape[1]))(sent_embed)
    print(np.shape(sent_embed))
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_embed)
    word_bigru = Reshape((sent_length, 256))(word_bigru)
    # print(np.shape(word_bigru))
    word_attention = Attention(sent_length)(word_bigru)
    sent_encode = Reshape((-1, sent_num))(word_attention)
    # sent_encode = Model(sentence_input, word_attention)
    #
    # doc_input = Input(shape=(sent_num, sent_length), dtype="int32")
    # doc_encode = TimeDistributed(sent_encode)(doc_input)
    sent_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_encode)
    doc_attention = Attention(sent_num)(sent_bigru)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(doc_attention)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(input, output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model



def get_han(sent_num, sent_length, embed_weight, mask_zero=False):
    sentence_input = Input(shape=(sent_length,), dtype="int32",name='word_input')
    embedding = Embedding(
        name= "embeding",
        input_dim = embed_weight.shape[0],
        weights = [embed_weight],
        output_dim = embed_weight.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )
    sent_embed = embedding(sentence_input)
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_embed)
    word_attention = Attention(sent_length)(word_bigru)
    sent_encode = Model(sentence_input, word_attention,name='sent_encode')

    doc_input = Input(shape=(sent_num, sent_length), dtype="int32",name="sent_input")
    doc_encode = TimeDistributed(sent_encode)(doc_input)
    sent_bigru = Bidirectional(GRU(128, return_sequences=True))(doc_encode)
    sent_attention = Attention(sent_num)(sent_bigru)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(sent_attention)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(doc_input, output, name='doc_encode')
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def get_word_char_cnn(word_len, char_len, word_embed_weight, char_embed_weight):
    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim =char_embed_weight.shape[0],
        weights =[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))
    word_feat = convs_block(trans_word, convs=[1,2,3,4,5], f=256, name="word_conv")
    char_feat = convs_block(trans_char, convs=[1,2,3,4,5], f=256, name="char_conv")
    feat = concatenate([word_feat, char_feat])
    dropfeat = Dropout(0.4)(feat) # 0.4
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat))) # 256
    output = Dense(4, activation="softmax")(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def get_word_char_hcnn(sent_num, sent_word_length, sent_char_length, word_embed_weight, char_embed_weight, mask_zero=False):
    sentence_word_input = Input(shape=(sent_word_length,), dtype="int32")
    word_embedding = Embedding(
        name = "word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim= word_embed_weight.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )
    sent_word_embed = word_embedding(sentence_word_input)
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_word_embed)
    word_attention = Attention(sent_word_length)(word_bigru)
    sent_word_encode = Model(sentence_word_input, word_attention)

    sentence_char_input = Input(shape=(sent_char_length,), dtype="int32")
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        mask_zero = mask_zero,
    )
    sent_char_embed = char_embedding(sentence_char_input)
    char_bigru = Bidirectional(GRU(64, return_sequences=True))(sent_char_embed)
    char_attention = Attention(sent_char_length)(char_bigru)
    sent_char_encode = Model(sentence_char_input, char_attention)

    review_word_input = Input(shape=(sent_num, sent_word_length), dtype="int32")
    review_word_encode = TimeDistributed(sent_word_encode)(review_word_input)
    review_char_input = Input(shape=(sent_num, sent_char_length),dtype="int32")
    review_char_encode = TimeDistributed(sent_char_encode)(review_char_input)
    review_encode = concatenate([review_word_encode, review_char_encode])
    unvec = convs_block(review_encode, convs=[1,2,3,4,5], f=256)
    dropfeat = Dropout(0.2)(unvec)
    fc = Activation(activation='relu')(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(4, activation="softmax")(fc)
    model = Model([review_word_input, review_char_input], output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accracy'])
    return model


def convs_block_v2(data, convs = [3,4,5], f=256, name="conv2_feat"):
    pools = []
    for c in convs:
        conv = Conv1D(f, c, activation='elu', padding='same')(data)
        conv = MaxPooling1D(c)(conv)
        conv = Conv1D(f, c-1, activation='elu', padding='same')(conv)
        conv = MaxPooling1D(c-1)(conv)
        conv = Conv1D(int(f/2), 2, activation='elu', padding='same')(conv)
        conv = MaxPooling1D(2)(conv)
        conv = Flatten()(conv)
        pools.append(conv)
    return concatenate(pools, name=name)


def resnet_convs_block(data, convs =[3,4,5], f=256, name="deep_conv_feat"):

    pools = []
    x_short = data
    for c in convs:
        conv = Conv1D(int(f/2), c, activation='relu', padding='same')(data)
        conv = MaxPooling1D(c)(conv)
        conv = Conv1D(f, c-1, activation='relu', padding='same')(conv)
        conv = MaxPooling1D(c-1)(conv)
        conv = Conv1D(f*2, 2, activation='relu', padding='same')(conv)
        conv = MaxPooling1D(2)(conv)
        conv = Flatten()(conv)
        pools.append(conv)
    conv_shocut = BatchNormalization()(Conv1D(filters=f, kernel_size=3, padding="valid")(x_short))
    # conv_shocut = GlobalMaxPooling1D()(conv_shocut)
    conv_shocut = MaxPooling1D(3)(conv_shocut)
    conv_shocut = Flatten()(conv_shocut)
    pools.append(conv_shocut)
    return Activation('relu')(concatenate(pools, name=name))


    # conv_shortcut = Conv1D(f,c=1, activation='relu')

def get_textcnn_v2(seq_length, embed_weight):
    content = Input(shape=(seq_length,), dtype='int32')
    embedding = Embedding(
        input_dim=embed_weight.shape[0],
        weights=[embed_weight],
        output_dim=embed_weight.shape[1],
        name="embedding",
        trainable=False)
    embed = embedding(content)
    embed = SpatialDropout1D(0.2)(embed)
    trans_content = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embed))))
    # unvec = convs_block_v2(trans_content)
    unvec = resnet_convs_block(trans_content)
    dropfeat = Dropout(0.4)(unvec)
    fc = Activation(activation='relu')(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(4, activation="softmax")(fc)
    # output = Dense(4, activation="sigmoid")(fc)
    model = Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def get_word_char_cnn_v2(word_len, char_len, word_embed_weight, char_embed_weight):
    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights = [char_embed_weight],
        output_dim = char_embed_weight.shape[1],
        trainable=False
    )

    word_embed = SpatialDropout1D(0.2)(word_embedding(word_input))
    char_embed = SpatialDropout1D(0.2)(char_embedding(char_input))

    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embed))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embed))))

    word_feat = convs_block_v2(trans_word, name='word_conv')
    char_feat = convs_block_v2(trans_char, name='char_conv')

    unvec = concatenate([word_feat, char_feat])
    dropfeat = Dropout(0.4)(unvec)
    fc = Activation(activation="relu")(BatchNormalization()(Dropout(0.2)(Dense(512)(dropfeat))))
    output = Dense(4, activation="softmax")(fc)
    model = Model(inputs=[word_input, char_input], outputs = output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def get_wordp_char_cnn_v2(word_len, char_len, word_embed_weight, char_embed_weight):
    word_input = Input(shape=(word_len,), dtype="int32")
    wordp_input = Input(shape=(word_len,), dtype='int32')
    char_input = Input(shape=(char_len,), dtype="int32")
    word_embedding = Embedding(
        name = "word_embeding",
        input_dim = word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable=False
    )
    wordp_embedding = Embedding(
        name="wordp_embedding",
        input_dim=57,
        output_dim=64
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )
    word_union = concatenate([word_embedding(word_input), wordp_embedding(wordp_input)])
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(320))(word_union))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))
    word_feat = convs_block_v2(trans_word)
    char_feat = convs_block_v2(trans_char)
    unvec = concatenate([word_feat, char_feat])
    dropfeat = Dropout(0.4)(unvec)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(512)(dropfeat)))
    output = Dense(4, activation="softmax")(Dropout(0.2)(fc))
    model = Model(inputs=[word_input, wordp_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


def get_capsule_gru(maxlen, embed_weight):

    Num_capsule = 10
    Dim_capsule = 20
    Routings = 5

    input = Input(shape=(maxlen,))
    embedding = Embedding(
        name = "embedding",
        input_dim = embed_weight.shape[0],
        weights = [embed_weight],
        output_dim = embed_weight.shape[1],
        trainable=False
    )
    embed_layer = SpatialDropout1D(0.2)(embedding(input))

    # bigru = Bidirectional(GRU(128, return_sequences=True))(embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(embed_layer)
    capsule = Bidirectional(GRU(128, return_sequences=True))(capsule)
    # capsule = Flatten()(capsule)
    avg_pool = GlobalAveragePooling1D()(capsule)
    max_pool = GlobalMaxPooling1D()(capsule)
    conc = concatenate([avg_pool, max_pool])
    dropfeat = Dropout(0.4)(conc)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_rcnn(maxlen, embed_weight):

    input = Input(shape=(maxlen,))
    embedding = Embedding(
        name = "embedding",
        input_dim = embed_weight.shape[0],
        weights = [embed_weight],
        output_dim = embed_weight.shape[1],
        trainable = False
    )
    embed_layer = SpatialDropout1D(0.2)(embedding(input))
    bigru = Bidirectional(GRU(128, return_sequences=True))(embed_layer)
    conv = Conv1D(128, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(bigru)
    avg_pool = GlobalAveragePooling1D()(conv)
    max_pool = GlobalMaxPooling1D()(conv)
    conc = concatenate([avg_pool, max_pool])
    dropfeat = Dropout(0.4)(conc)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(64)(dropfeat)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_cgru(maxlen, embed_weight):

    input = Input(shape=(maxlen,))
    embedding = Embedding(
        name = "embedding",
        input_dim = embed_weight.shape[0],
        weights = [embed_weight],
        output_dim= embed_weight.shape[1],
        trainable = False
    )
    # embed_layer = SpatialDropout1D(0.2)(embedding(input))
    embed_layer = (embedding(input))
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embed_layer))))
    conv = Conv1D(filters=128, kernel_size=3, activation='elu', padding='valid')(trans_word)
    bigru = Bidirectional(GRU(128, return_sequences=True))(conv)
    avg_pool = GlobalAveragePooling1D()(bigru)
    max_pool = GlobalMaxPooling1D()(bigru)
    conc = concatenate([avg_pool, max_pool])
    dropfeat = Dropout(0.4)(conc)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_word_char_rnn(word_len, char_len, word_embed_weight, char_embed_weight):
    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim =char_embed_weight.shape[0],
        weights =[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    # word_bigru2 = Bidirectional(GRU(128, return_sequences=True))(word_bigru)
    avg_pool = GlobalAveragePooling1D()(word_bigru)
    max_pool = GlobalMaxPooling1D()(word_bigru)
    word_feat = concatenate([avg_pool, max_pool], axis=1)

    char_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_char)
    # char_bigru2 = Bidirectional(GRU(128, return_sequences=True))(char_bigru)
    avg_pool = GlobalAveragePooling1D()(char_bigru)
    max_pool = GlobalMaxPooling1D()(char_bigru)
    char_feat = concatenate([avg_pool, max_pool], axis=1)
    feat = concatenate([word_feat, char_feat], axis=1)
    dropfeat = Dropout(0.3)(feat) # 0.4
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat))) # 256
    output = Dense(4, activation="softmax")(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def get_word_char_rcnn(word_len, char_len, word_embed_weight, char_embed_weight):

    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")
    word_embedding = Embedding(
        name = "word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable = False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))
    # trans_word = word_embedding(word_input)
    # trans_char = char_embedding(char_input)

    bigru_word = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    conv_word = convs_block(bigru_word,convs=[1,2,3], name='conv_word')

    bigru_char = Bidirectional(GRU(128, return_sequences=True))(trans_char)
    conv_char = convs_block(bigru_char, convs=[1,2,3], name='conv_char')

    conc = concatenate([conv_word, conv_char])
    dropfeat = Dropout(0.4)(conc)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_word_char_capsule_gru(word_len, char_len, word_embed_weight, char_embed_weight):

    Num_capsule = 10
    Dim_capsule = 20
    Routings = 5

    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")
    word_embedding = Embedding(
        name = "word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable=False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )

    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))

    word_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    char_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_char)

    word_capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(word_bigru)
    char_capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(char_bigru)

    word_capsule = Flatten()(word_capsule)
    char_capsule = Flatten()(char_capsule)

    conc = concatenate([word_capsule, char_capsule])
    dropfeat = Dropout(0.4)(conc)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_word_rcnn_char_rnn(word_len, char_len, word_embed_weight, char_embed_weight):

    word_input = Input(shape=(word_len, ), dtype="int32")
    char_input = Input(shape=(char_len, ), dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable = False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim = char_embed_weight.shape[0],
        weights = [char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable = False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))

    bigru_word = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    word_feat = convs_block(bigru_word, convs=[1,2,3], name='conv_word')

    char_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_char)
    avg_pool = GlobalAveragePooling1D()(char_bigru)
    max_pool = GlobalMaxPooling1D()(char_bigru)
    char_feat = concatenate([avg_pool, max_pool], axis=1)

    conc = concatenate([word_feat, char_feat])
    dropfeat =  Dropout(0.4)(conc)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_word_rnn_char_rcnn(word_len, char_len, word_embed_weight, char_embed_weight):

    word_input = Input(shape=(word_len, ), dtype="int32")
    char_input = Input(shape=(char_len, ), dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable = False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim = char_embed_weight.shape[0],
        weights = [char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable = False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))

    # rnn
    bigru_word = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    avg_pool = GlobalAveragePooling1D()(bigru_word)
    max_pool = GlobalMaxPooling1D()(bigru_word)
    word_feat = concatenate([avg_pool, max_pool], axis=1)

    # rcnn
    char_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_char)
    char_feat = convs_block(char_bigru, convs=[1, 2, 3], name='conv_char')


    conc = concatenate([word_feat, char_feat])
    dropfeat =  Dropout(0.4)(conc)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_word_rcnn_char_cgru(word_len, char_len, word_embed_weight, char_embed_weight):
    word_input = Input(shape=(word_len, ), dtype="int32")
    char_input = Input(shape=(char_len, ), dtype="int32")
    word_embedding = Embedding(
        name = "word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable = False
    )
    char_embedding = Embedding(
        name = "char_embedding",
        input_dim = char_embed_weight.shape[0],
        weights = [char_embed_weight],
        output_dim = char_embed_weight.shape[1],
        trainable = False
    )

    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))

    # word rcnn
    bigru_word = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    word_feat = convs_block(bigru_word, convs=[1,2,3], name='conv_word')

    # char cgru
    conv_char = (BatchNormalization()(Conv1D(filters=256, kernel_size=3, padding="valid")(trans_char)))
    bigru_char = Bidirectional(GRU(128, return_sequences=True))(conv_char)
    avg_pool = GlobalAveragePooling1D()(bigru_char)
    max_pool = GlobalMaxPooling1D()(bigru_char)
    char_feat = concatenate([avg_pool, max_pool], axis=1)

    feat = concatenate([word_feat, char_feat], axis=1)
    dropfeat = Dropout(0.3)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(4, activation="softmax")(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_word_cgru_char_rcnn(word_len, char_len, word_embed_weight, char_embed_weight):

    word_input = Input(shape=(word_len, ), dtype="int32")
    char_input = Input(shape=(char_len, ), dtype="int32")
    word_embedding = Embedding(
        name = "word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable = False
    )
    char_embedding = Embedding(
        name = "char_embedding",
        input_dim = char_embed_weight.shape[0],
        weights = [char_embed_weight],
        output_dim = char_embed_weight.shape[1],
        trainable = False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))\

    # word cgru
    conv_word = (BatchNormalization()(Conv1D(filters=256, kernel_size=3, padding="valid")(trans_word)))
    bigru_word = Bidirectional(GRU(128, return_sequences=True))(conv_word)
    avg_pool = GlobalAveragePooling1D()(bigru_word)
    max_pool = GlobalMaxPooling1D()(bigru_word)
    word_feat = concatenate([avg_pool, max_pool], axis=1)

    # char rcnn
    bigru_char = Bidirectional(GRU(128, return_sequences=True))(trans_char)
    char_feat = convs_block(bigru_char, convs=[1,2,3], name="conv_char")

    feat = concatenate([word_feat, char_feat], axis=1)
    dropfeat = Dropout(0.4)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_word_char_cnn_fe(word_len, char_len, fe_len, word_embed_weight, char_embed_weight):

    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")
    feature_input = Input(shape=(fe_len, ), dtype="int32")

    word_embedding = Embedding(
        name="word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable = False
    )
    char_embedding = Embedding(
        name = "char_embedding",
        input_dim = char_embed_weight.shape[0],
        weights = [char_embed_weight],
        output_dim = char_embed_weight.shape[1],
        trainable = False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))

    word_feat = convs_block(trans_word, convs=[1,2,3,4,5], f=256, name="word_conv")
    char_feat = convs_block(trans_char, convs=[1,2,3,4,5], f=256, name="char_conv")
    feat = concatenate([word_feat, char_feat])
    dropfeat = Dropout(0.4)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    fc = concatenate([fc, feature_input])
    output = Dense(4, activation="softmax")(fc)
    model = Model(inputs=[word_input, char_input, feature_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accruracy'])
    return model