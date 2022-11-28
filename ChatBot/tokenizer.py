
from ChatBot.utils import is_string,do_lower_reg
from Basic_Layer_Models.Transformer.backend.snippets import truncate_sequences
from tqdm import tqdm
import unicodedata,jieba
import numpy as np

class Tokenizer:
    def __init__(self,token_dict,pre_token=None,
                 init_tokens = ['[PAD]', '[STA]','[END]','[UNK]'],
                 do_lower_case = False,
                 max_len = None):
        self._do_lower_case = do_lower_case
        self.pre_token = pre_token
        self.max_len = max_len
        if is_string(token_dict):
            token_dict = self.load_vocab(token_dict,init_tokens)
        self.token_dict = token_dict
        for token in init_tokens:
            token_low = do_lower_reg(token)
            try:
                _token_id = token_dict[token]
                setattr(self, '_token_%s' % token_low, token)
                setattr(self, '_token_%s_id' % token_low, _token_id)
            except:
                pass
        self.token_dict_inv = {v: k for k, v in self.token_dict.items()}
        self.vocab_size = len(self.token_dict)

    def load_vocab(self,dict_path,init_tokens, encoding='utf-8'):
        token_list = []
        token_dict = {}
        with open(dict_path, encoding=encoding) as fr:
            for line in fr:
                token = line.split()
                token = token[0] if token else line.strip()
                token_list.append(token)
        for item in reversed(init_tokens):
            if item not in token_list:
                token_list.insert(0,item)
        for token in token_list:
            token_dict[token] = len(token_dict)
        return token_dict

    def encode(self,encode_text,target_flg = False):
        if not isinstance(encode_text,str):
            raise ValueError("encode_text should be a list")

        tokens = self._tokenize(encode_text)

        if target_flg:
            tokens.insert(0,self._token_sta)

        if self.max_len is not None:
            if self.max_len < 2:
                self.max_len = 2
            tokens = tokens[:self.max_len-1]

        tokens.append(self._token_end)
        token_ids = [self.token_to_id(token) for token in tokens]
        return token_ids

    '''
        碰到end或者连续的pad则返回结果
    '''
    def decoder(self,decode_index):
        res = ''
        if not (isinstance(decode_index,list) or isinstance(decode_index,np.ndarray)):
            raise ValueError("decode_index should be a list")
        for index,id in enumerate(decode_index):
            word = self.id_to_token(id)
            if word == self._token_end:
                return res
            if not index == len(decode_index) and (id==self._token_pad_id
                                                   and decode_index[index+1]==self._token_pad_id):
                return res
            if word not in [self._token_sta,self._token_unk,self._token_pad]:
                res+=word


        return res

    def _tokenize(self, text, pre_tokenize=True):
        """基本分词函数
        """
        if self._do_lower_case:
            text = text.lower()
            text = unicodedata.normalize('NFD', text)
            text = ''.join([
                ch for ch in text if unicodedata.category(ch) != 'Mn'
            ])

        if pre_tokenize and self.pre_token is not None:
            tokens = []
            for token in self.pre_token(text):
                if token in self.token_dict:
                    tokens.append(token)
                else:
                    if token.isdigit():
                        tokens.append(self._token_unk)
                    else:
                        tokens.extend(self._tokenize(token,False))
            return tokens

        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch

        tokens = []
        for word in spaced.strip().split():
            tokens.extend(word)

        return tokens


    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
            unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        提醒：unicodedata.category这个函数在py2和py3下的
        表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
        在py3下的结果是'Po'。
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126 or \
            unicodedata.category(ch).startswith('P')

    @staticmethod
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002'

    @staticmethod
    def _is_cjk_character(ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def token_to_id(self,token):
        '''
        :param token:
        :return:
        '''
        return self.token_dict.get(token,self._token_unk_id)
    def id_to_token(self,id):
        '''
        :param id:
        :return:
        '''
        return self.token_dict_inv.get(id,self._token_unk)



if __name__ == '__main__':
    def pre_token(text):
        return jieba.lcut(text)

    dict_path = './processed_data/vocab_pure.txt'
    t2 = Tokenizer(dict_path, pre_token=pre_token,max_len=15)
    text = "小黄鸭243633699,你有女朋友么"
    encode = t2.encode(text,target_flg=False)
    decode = t2.decoder(encode)
    print(encode)
    print(decode)
