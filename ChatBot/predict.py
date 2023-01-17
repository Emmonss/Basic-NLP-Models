import json,os
from ChatBot import train_config
from ChatBot.utils import get_max_from_list
from ChatBot.sentToken import DialogCodeTrans
from BasicLayerModels.RNNs.models.SequenceAttention import Seq2SeqAttention
from pprint import pprint
from past.builtins import raw_input

def get_model(type_name,model_name,root_path='./models'):
    #默认取当前目录下最新的模型
    time_path = get_max_from_list(os.listdir(os.path.join(root_path,type_name)))
    print(time_path)
    with open(os.path.join(root_path,type_name,time_path,train_config.params_save_name),'r',encoding='utf=8') as fr:
        params = json.load(fr)
        vocab_size = params['vocab_size']
        embed_dim = params['embed_dim']
        max_sent_len = params['sent_maxlen']
        hidden_units = params['hidden_units']
        lr = params['lr']
        dict_path = params['dict_path']
        pretoken_flg = params['pretoken_flg']

    dct = DialogCodeTrans(dict_path=dict_path,
                                sent_maxlen=max_sent_len,
                                pre_token_flag=pretoken_flg)
    dct.predict_encode("test")
    model = Seq2SeqAttention(embeding_dim=embed_dim,
                                 max_sent_len=max_sent_len,
                                 encoder_units=hidden_units,
                                 lr=lr,
                                mode='pre',
                                tokenizer=dct.tokenizer)
    model.load_weights(model_path=os.path.join(root_path,type_name,time_path),
                        model_name=model_name)
    print("done!")
    return dct,model

def main():
    type_name = 'xhj'
    model_name = train_config.model_name_xhj
    dct, bot = get_model(type_name,model_name)
    while True:
        text = raw_input('you:')
        encode = dct.predict_encode(text)
        print(encode)
        _, decoder = bot.evaluate(encode)
        print(decoder)
        output = dct.idx_2_sent(decoder)[0]
        print(output)
if __name__ == '__main__':
    main()
    # get_model(type_name,model_name)
    pass