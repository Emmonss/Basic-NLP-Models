import json,os
from ChatBot import train_config
from ChatBot.utils import get_max_from_list
from ChatBot.sentToken import DialogCodeTrans
from Basic_Layer_Models.RNNs.models.SequenceAttention import Seq2SeqAttention
from pprint import pprint
from past.builtins import raw_input

def get_model(type_name,model_name,root_path='./models'):
    #默认取当前目录下最新的模型
    time_path = get_max_from_list(os.listdir(os.path.join(root_path,type_name)))
    with open(os.path.join(root_path,type_name,time_path,train_config.params_save_name),'r',encoding='utf=8') as fr:
        params = json.load(fr)
        vocab_size = params['vocab_size']
        embed_dim = params['embed_dim']
        max_sent_len = params['sent_maxlen']
        hidden_units = params['hidden_units']
        lr = params['lr']
        dict_path = params['dict_vocab_pure']
        pretoken_flg = params['pretoken_flg']

    tokenizer = DialogCodeTrans(dict_path=dict_path,
                                sent_maxlen=max_sent_len,
                                pre_token_flag=pretoken_flg)
    tokenizer.predict_encode("test")
    model = Seq2SeqAttention(vocab_size=vocab_size,
                                 embeding_dim=embed_dim,
                                 max_sent_len=max_sent_len,
                                 encoder_units=hidden_units,
                                 lr=lr,
                                mode='pre'
                                 )
    model.load_weights(model_path=os.path.join(root_path,type_name,time_path),
                        model_name=model_name)
    print("done!")
    return tokenizer,model

def main():
    type_name = 'xhj'
    model_name = train_config.model_name_xhj
    tokenizer, bot = get_model(type_name,model_name)
    while True:
        text = raw_input('you:')
        encode = tokenizer.predict_encode(text)
        _, decoder = bot.evaluate(encode)
        output = tokenizer.idx_2_sent(decoder)[0]
        print(output)
if __name__ == '__main__':
    main()
    # get_model(type_name,model_name)
    pass