import os

from Basic_Layer_Models.RNNs.utils.Tokenize import Tokenizer
from ChatBot.utils import pre_token,read_cropus
from tensorflow.keras.preprocessing.sequence import pad_sequences

import jieba
jieba.load_userdict('./processed_data/jieba_dict_pure.txt')

def pre_token(text):
    return jieba.lcut(text)

class DialogCodeTrans:
    def __init__(self,dict_path,sent_maxlen=50,pre_token_flag=True):
        self.dict_path = dict_path
        self.pre_token_flag = pre_token_flag
        self.sent_maxlen = sent_maxlen
        self._get_tokenizer()

    def _get_tokenizer(self):
        if self.pre_token_flag:
            self.tokenizer = Tokenizer(token_dict=self.dict_path, do_lower_case=False, pred_tokenizer=pre_token)
        else:
            self.tokenizer = Tokenizer(token_dict=self.dict_path, do_lower_case=False)
        self.start_id = self.tokenizer._token_dict['[STA]']
        self.end_id = self.tokenizer._token_dict['[END]']

    def _sentence_encode(self,encode_text,decode_text):
        token_encoder,seg_encoder = self.tokenizer.encode(encode_text)
        token_decoder,seg_decoder = self.tokenizer.encode(decode_text)
        token_encoder = token_encoder+[self.end_id]
        token_decoder = [self.start_id]+token_decoder+[self.end_id]
        return token_encoder,token_decoder

    def _segment_decode(self,token_encoder,token_decoder):
        encode_text = self.tokenizer.id_to_tokens(token_encoder)
        decode_text = self.tokenizer.id_to_tokens(token_decoder)
        return encode_text,decode_text

    def get_sent_to_array(self,input_path,target_path):
        if not os.path.exists(input_path) or os.path.exists(target_path):
            raise ValueError("the dialog path is not exists")
        token_encoder_list, token_decoder_list = [],[]
        input_list = read_cropus(input_path)
        target_list = read_cropus(target_path)
        assert len(input_path)==len(target_path),"the length of input and target should be the same"
        for i,encode_text in enumerate(input_list):
            decode_text = target_list[i]
            token_encoder, token_decoder = self._sentence_encode(encode_text,decode_text)
            token_encoder_list.append(token_encoder)
            token_decoder_list.append(token_decoder)
            token_encoder_list = pad_sequences(token_encoder_list, value=0, padding='post', maxlen=self.sent_maxlen)
            token_decoder_list = pad_sequences(token_decoder_list, value=0, padding='post', maxlen=self.sent_maxlen + 1)


if __name__ == '__main__':
    dict_path = './processed_data/dt_dict.txt'
    pre_token_flag=True
    sent_maxlen = 50
    print(jieba.lcut("你是12312431傻逼吗"))
    tokenizer = DialogCodeTrans(dict_path,sent_maxlen,pre_token_flag)
    token_encoder,token_decoder =tokenizer._sentence_encode("你是12312431傻逼吗","你才是傻逼")
    encode_text,decode_text = tokenizer._segment_decode(token_encoder,token_decoder)
    print(token_encoder,token_decoder)
    print(encode_text,decode_text)
    pass
