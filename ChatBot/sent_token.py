import os

from ChatBot.tokenizer import Tokenizer
from ChatBot.utils import pre_token,read_cropus
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba

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
            self.tokenizer = Tokenizer(token_dict=self.dict_path,max_len=self.sent_maxlen, pre_token=pre_token)
        else:
            self.tokenizer = Tokenizer(token_dict=self.dict_path,max_len=self.sent_maxlen)

    def _sentence_encode(self,encode_text,decode_text):
        token_encoder = self.tokenizer.encode(encode_text)
        token_decoder = self.tokenizer.encode(decode_text,target_flg=True)
        return token_encoder,token_decoder

    def _segment_decode(self,token_list):
        encode_text = self.tokenizer.decoder(token_list)
        return encode_text

    def get_sent_to_array(self,input_path,target_path):
        if not os.path.exists(input_path) or not os.path.exists(target_path):
            raise ValueError("the dialog path is not exists")
        token_input_list, token_target_list = [],[]
        input_list = read_cropus(input_path)
        target_list = read_cropus(target_path)
        assert len(input_list)==len(target_list),"the length of input and target should be the same"

        for i,encode_text in enumerate(input_list):
            decode_text = target_list[i]
            token_encoder, token_decoder = self._sentence_encode(encode_text,decode_text)
            token_input_list.append(token_encoder)
            token_target_list.append(token_decoder)
        token_input_list = pad_sequences(token_input_list, value=self.tokenizer._token_pad_id,
                                           padding='post', maxlen=self.sent_maxlen)
        token_target_list = pad_sequences(token_target_list, value=self.tokenizer._token_pad_id,
                                           padding='post', maxlen=self.sent_maxlen + 1)

        return token_input_list,token_target_list
        # print(token_input_list)
        # print(token_target_list)
        # res_encode = [self._segment_decode(item) for item in token_input_list]
        # res_target = [self._segment_decode(item) for item in token_target_list]
        # for i in range(len(res_encode)):
        #     print('='*10)
        #     print("input:{}".format(res_encode[i]))
        #     print("target:{}".format(res_target[i]))




if __name__ == '__main__':
    dict_path = './processed_data/vocab_pure.txt'

    in_path = './cropus/di/input_test.txt'
    ou_path = './cropus/di/output_test.txt'
    sent_max_len = 10
    t = DialogCodeTrans(dict_path=dict_path,sent_maxlen=sent_max_len)
    t.get_sent_to_array(in_path,ou_path)
    pass
