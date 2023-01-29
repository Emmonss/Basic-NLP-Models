
import os,json
import numpy as np
from ChatBot.sentToken import DialogCodeTrans
from BasicLayerModels.RNNs.models.SequenceAttention import Seq2SeqAttention
from ChatBot import trainConfig
from ChatBot.utils import make_dict
from datetime import datetime
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as e:
        print(e)





def train_xhj():
    inpath = trainConfig.inpath_xhj
    tarpath = trainConfig.tarpath_xhj
    save_path = trainConfig.save_model_path_xhj
    model_name = trainConfig.model_name_xhj

    if trainConfig.dict_path is not None:
        print("make self dict")
        make_dict(input_list=[inpath,tarpath],
                  output_list=trainConfig.dict_path)
        dict_path = trainConfig.dict_path
    else:
        dict_path = trainConfig.dict_vocab_pure

    dct = DialogCodeTrans(dict_path=dict_path,
                          sent_maxlen=trainConfig.sent_maxlen,
                          pre_token_flag=trainConfig.pretoken_flg)
    #
    print("loading dialog and encoding")
    token_input_list, token_target_list = dct.sent_2_idx(inpath, tarpath)
    print("input token shape:{}".format(np.shape(token_input_list)))
    print("target token shape:{}".format(np.shape(token_target_list)))
    model = Seq2SeqAttention(embeding_dim=trainConfig.embed_dim,
                             max_sent_len=trainConfig.sent_maxlen,
                             encoder_units=trainConfig.hidden_units,
                             lr=trainConfig.lr,
                             teach_forcing_ran=trainConfig.teach_forcing_ran,
                             tokenizer=dct.tokenizer)
    print("training......")
    model.fit_val(X=[token_input_list, token_target_list],
                  Y=token_target_list[:, 1:],
                  epoch=trainConfig.epoch,
                  batch_size=trainConfig.batch_size)
    print("done!")

    if trainConfig.saving_flg:
        save_path = save_path.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_weights(model_path=save_path, model_name=model_name)
        basic_params = {
            'vocab_size': dct.vocab_size,
            'embed_dim': trainConfig.embed_dim,
            'sent_maxlen': trainConfig.sent_maxlen,
            'hidden_units': trainConfig.hidden_units,
            'lr': trainConfig.lr,
            'epoch': trainConfig.epoch,
            'batch_size': trainConfig.batch_size,
            'pretoken_flg': trainConfig.pretoken_flg,
            'dict_path': dict_path,
            'inpath': inpath,
            'tarpath': tarpath
        }
        with open('{}/{}'.format(save_path, trainConfig.params_save_name), 'w', encoding='utf-8') as fw:
            json.dump(basic_params, fw, ensure_ascii=False)

if __name__ == '__main__':
    train_xhj()
    pass