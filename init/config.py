# coding=utf-8
import pickle

class Config():

    data_dir = '/media/iiip/Elements/smp/new_data'
    cache_dir = data_dir + '/cache'

    # 训练集的样本
    train_org_path = data_dir + '/train_data/training_hanlp_cut.txt'
    train_all_data_path = data_dir + '/train_data/all.csv'
    train_data_path = data_dir + '/train_data/train.csv'
    test_data_path = data_dir + '/train_data/test.csv'

    # 验证集的样本
    vali_org_path = data_dir + '/validation_data/validation_hanlp_cut.txt'
    vali_id_path = data_dir + '/validation_data/validation_id.txt'
    vali_data_path = data_dir + '/validation_data/validation.csv'

    # 最终集的样本
    final_org_path = data_dir + '/final_data/testing_hanlp_cut.txt'
    final_id_path = data_dir + '/final_data/testing_id.txt'
    final_data_path = data_dir + '/final_data/testing.csv'
    final_seq_path = data_dir + '/final_data/testing_seq.pk'

    # 预先的词向量
    open_w2v_path = "/media/iiip/数据/smp_data/词向量/Hanlp_cut_news.bin"

    # 词dict及预训练的权重
    word_vocab_path = cache_dir + '/word_vocab.pk'
    word_embed_path = cache_dir + "/word_embed.pk"

    word_vocab_v2_path = data_dir + '/new_vocab/word_vocab_v2.pk'
    word_embed_v2_path = data_dir + '/new_vocab/word_embed_v2.pk'

    # 字dict及预训练的权重
    char_vocab_path = cache_dir + '/char_vocab.pk'
    char_embed_path = cache_dir + '/char_embed.pk'

    char_vocab_v2_path = data_dir + '/new_vocab/char_vocab_v2.pk'
    char_embed_v2_path = data_dir + '/new_vocab/char_embed_v2.pk'

    # 截断补齐句子个数, 句子词的个数, 句子字的个数(HAN)
    sentence_num = 35
    sentence_word_length = 70
    sentence_char_length = 100

    # 截断补齐文本词的个数, 字的个数
    word_seq_maxlen = 700
    char_seq_maxlen = 990

    # 模型路径
    textcnn_model_path = cache_dir + '/cnn/dp_embed_word_cnn_epoch_9_[0.97035294 0.99581006 0.99807074 0.97908565].h5'
    # word_char_cnn_path = cache_dir + '/word_char/dp_embed_word_char_cnn_epoch_10_[ 0.97720577  0.99609919  0.99855468  0.98343246].h5'
    word_char_cnn_path = cache_dir + '/word_char/dp_embed_word_char_cnn_epoch_9_[ 0.97526832  0.99665365  0.99855515  0.98241525].h5'
    charcnn_model_path = cache_dir + '/char_cnn/dp_embed_char_cnn_epoch_9_[ 0.97138414  0.99720904  0.99871465  0.9805291 ].h5'
    deep_model_path = cache_dir + '/deep_cnn/dp_embed_deep_cnn_epoch_13_[ 0.97330022  0.99456294  0.99903692  0.98058355].h5'
    rcnn_model_path = cache_dir + '/rcnn/dp_embed_word_rcnn_epoch_10_[ 0.97010277  0.99385646  0.99823293  0.97790931].h5'
    deep_word_char_model_path = cache_dir + '/deep_word_char/dp_embed_deep_word_char_cnn_epoch_8_[ 0.97857031  0.99637378  0.99807445  0.98385224].h5'
    cgru_model_path = cache_dir + '/c_gru/dp_embed_c_gru_epoch_9_[ 0.96931942  0.99525007  0.99775425  0.97916887].h5'
    # word_char_rnn_path = cache_dir + '/word_char_rnn/dp_embed_word_char_cnn_epoch_12_[ 0.97908211  0.99580537  0.99871589  0.98439148].h5'
    word_char_rnn_path = cache_dir + '/word_char_rnn/dp_embed_word_char_cnn_epoch_11_[ 0.97945205  0.99692737  0.99887658  0.98529878].h5'
    # word_char_rcnn_path = cache_dir + '/word_char_rcnn/dp_embed_word_char_rcnn_epoch_8_[ 0.97859709  0.99692823  0.99807384  0.98406669].h5'
    # word_char_rcnn_path = cache_dir + '/word_char_rcnn/dp_embed_word_char_rcnn_epoch_6_[ 0.9787234   0.99623378  0.99855422  0.98463495].h5'
    word_char_rcnn_path = cache_dir + '/word_char_rcnn/best/dp_embed_word_char_rcnn_epoch_7_[ 0.97914072  0.99637277  0.99855468  0.98381124].h5' # 验证集最好的模型, batch:64 --- BN

    # 模型列表
    # cnn 结构
    word_char_cnn_v2_list = [
        cache_dir + '/word_char_cnn_v2/' + 'dp_embed_word_char_cnn_epoch_13_[ 0.9784863   0.99637277  0.99887586  0.98470363].h5',
        # cache_dir + '/word_char/best/' + '64/dp_embed_word_char_cnn_epoch_9_[ 0.97748098  0.99679264  0.99823463  0.98356135].h5',
        # cache_dir + '/word_char/best/' + '64/dp_embed_word_char_cnn_epoch_11_[ 0.97732215  0.9958194   0.99871465  0.9836726 ].h5'
    ]
    # rnn 结构
    word_char_rnn_list = [
        cache_dir + '/word_char_rnn/best/' +'dp_embed_word_char_cnn_epoch_7_[ 0.97792975  0.99706745  0.99871589  0.98401271].h5',
        cache_dir + '/word_char_rnn/best/' + 'dp_embed_word_char_cnn_epoch_11_[ 0.97945205  0.99692737  0.99887658  0.98529878].h5',
        cache_dir + '/word_char_rnn/best/' + 'dp_embed_word_char_cnn_epoch_12_[ 0.97908211  0.99580537  0.99871589  0.98439148].h5'
    ]
    # rcnn 结构
    word_char_rcnn_list = [
        cache_dir + '/word_char_rcnn/best/' + 'dp_embed_word_char_rcnn_epoch_7_[ 0.97914072  0.99637277  0.99855468  0.98381124].h5',
        cache_dir + '/word_char_rcnn/best/' + 'dp_embed_word_char_rcnn_epoch_8_[ 0.97859709  0.99692823  0.99807384  0.98406669].h5',
        cache_dir + '/word_char_rcnn/best/' + 'dp_embed_word_char_rcnn_epoch_9_[ 0.97825066  0.99595819  0.99823293  0.98322608].h5'
    ]

    # # capsule-gru 结构
    capsule_gru_list = [
        cache_dir + '/capsule_gru/best/' + 'dp_embed_word_char_capsule_gru_epoch_7_[ 0.96771232  0.99609157  0.99935774  0.97677406].h5',
        cache_dir + '/capsule_gru/best/' + 'dp_embed_word_char_capsule_gru_epoch_8_[ 0.9697156   0.99580537  0.9990363   0.97837435].h5',
        cache_dir + '/capsule_gru/best/' + 'dp_embed_word_char_capsule_gru_epoch_9_[ 0.96674802  0.99594349  0.99935733  0.97634086].h5'
    ]


    # word_rnn_char_rcnn 结构
    word_rnn_char_rcnn_list = [
        cache_dir + '/word_rnn_char_rcnn_v2/best/' + 'dp_embed_word_rnn_char_rcnn_epoch_6_[ 0.97730805  0.99679443  0.9988755   0.98367953].h5',
        cache_dir + '/word_rnn_char_rcnn_v2/best/' + 'dp_embed_word_rnn_char_rcnn_epoch_7_[ 0.97801341  0.99693252  0.99903599  0.98446253].h5',
        cache_dir + '/word_rnn_char_rcnn_v2/best/' + 'dp_embed_word_rnn_char_rcnn_epoch_8_[ 0.97709924  0.99651471  0.99871507  0.98391875].h5'
    ]

    # deep_word_char_cnn 结构
    word_char_cnn_list = [
        cache_dir + '/word_char_cnn_v2/best/' + 'dp_embed_word_char_cnn_epoch_10_[ 0.97758541  0.99637277  0.99839435  0.98394471].h5',
        cache_dir + '/word_char_cnn_v2/best/' + 'dp_embed_word_char_cnn_epoch_11_[ 0.97817     0.99637378  0.9988755   0.98406967].h5',
        cache_dir + '/word_char_cnn_v2/best/' + 'dp_embed_word_char_cnn_epoch_12_[ 0.97781914  0.99637074  0.99823407  0.9845175 ].h5'
    ]

    # word_rcnn_char_cgru 结构
    word_rcnn_char_cgru_list = [
        cache_dir + '/word_rcnn_char_cgru/best/' + 'dp_embed_word_rnn_char_rcnn_epoch_10_[ 0.98287671  0.99693337  0.9991973   0.98655088].h5',
        cache_dir + '/word_rcnn_char_cgru/best/'+ 'dp_embed_word_rnn_char_rcnn_epoch_9_[ 0.98218435  0.99665459  0.9991973   0.98618785].h5',
        cache_dir + '/word_rcnn_char_cgru/best/' + 'dp_embed_word_rnn_char_rcnn_epoch_6_[ 0.98113796  0.99706745  0.99951823  0.98573994].h5'
    ]

    # word_cgru_char_rcnn 结构
    word_cgru_char_rcnn_list = [
        cache_dir + '/word_cgru_char_rcnn/best/' + 'dp_embed_word_cgru_char_rcnn_epoch_7_[ 0.98103823  0.99665365  0.99903692  0.98516006].h5',
        cache_dir + '/word_cgru_char_rcnn/best/' + 'dp_embed_word_cgru_char_rcnn_epoch_9_[ 0.98070339  0.99665365  0.99855515  0.9850683 ].h5',
        cache_dir + '/word_cgru_char_rcnn/best/' + 'dp_embed_word_cgru_char_rcnn_epoch_10_[ 0.98089172  0.99595931  0.99887586  0.98462191].h5'
    ]

    #
    word_rcnn_char_rnn_list = [
        cache_dir + '/word_rcnn_char_rnn_v2/' + 'dp_embed_word_rcnn_char_rnn_epoch_7_[ 0.97945845  0.99665085  0.99919705  0.9857158 ].h5',
        cache_dir + '/word_rcnn_char_rnn_v2/' + 'dp_embed_word_rcnn_char_rnn_epoch_8_[ 0.97908864  0.99693337  0.99887658  0.98487893].h5',
        cache_dir + '/word_rcnn_char_rnn_v2' + 'dp_embed_word_rcnn_char_rnn_epoch_9_[ 0.97675149  0.99720904  0.9990363   0.98317914].h5'
    ]

    # fasttext word test& valid
    fasttext_word_test_pro = cache_dir + '/fasttext/word/word_test.pk'
    fasttext_char_test_pro = cache_dir + '/fasttext/char/char_test.pk'
    fasttext_word_valid_pro = cache_dir + '/fasttext/word/word_valid.pk'
    fasttext_char_valid_pro = cache_dir + '/fasttext/char/char_valid.pk'

    # validation 提交文件
    validation_submit_path = data_dir + '/submit/result.csv'
    final_submit_path = data_dir + '/submit/final.csv'


