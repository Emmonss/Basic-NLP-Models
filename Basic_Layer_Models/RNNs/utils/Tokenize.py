'''
    Bert4Keras中tokenizer的拙劣模仿
'''
from bert4keras import tokenizers

import jieba,re
import unicodedata
from Basic_Layer_Models.RNNs.utils.snippets import truncate_sequences,is_string,is_py2

def load_vocab(dict_path, encoding='utf-8', simplified=False,startwith=None):
    token_dict = {}
    with open(dict_path,encoding=encoding) as fr:
        for line in fr:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)

    if simplified:
        new_token_dict, keep_tokens = {}, []
        startwith = startwith or []
        for t in startwith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])

        for t,_ in sorted(token_dict.items(),key=lambda s:s[1]):
            if t not in new_token_dict:
                keep = True
                if len(t)>1:
                    for c in Tokenizer.stem(t):
                        if (Tokenizer._is_cjk_character(c) or Tokenizer._is_punctuation(c)):
                            keep=False
                            break
                if keep:
                    new_token_dict[t] = len(new_token_dict)
                    keep_tokens.append(token_dict[t])

        return new_token_dict,keep_tokens
    else:
        return token_dict

def save_vocab(dict_path,token_dict,encoding='utf-8'):
    with open(dict_path,'w',encoding=encoding) as fw:
        for k,v in sorted(token_dict.items(),key= lambda s:s[1]):
            fw.write(k+'\n')
    fw.close()

class TokenizerBase(object):
    '''
        分词器基类
    '''
    def __init__(self,
                 token_start='[CLS]',
                 token_end='[SEP]',
                 pred_tokenizer=None,
                 token_translate=None):
        '''
        :param token_start:
        :param token_end:
        :param pred_tokenizer:
        :param token_translate:
        '''
        self._token_pad = '[PAD]'
        self._token_mask = '[MASK]'
        self._token_unk = '[UNK]'
        self._token_start = token_start
        self._token_end = token_end
        self._pre_tokenizer = pred_tokenizer
        self._token_translate = token_translate or {}
        self._token_translate_inv = {
            v: k for k, v in self._token_translate.items()
        }
    def tokenize(self,text,maxlen=None,add_start=False,add_end=True):
        tokens = [self._token_translate.get(token) or token
                  for token in self._tokenize(text)]
        if self._token_start is not None and add_start:
            tokens.insert(0,self._token_start)
        if self._token_end is not None and add_end:
            tokens.append(self._token_start)

        if maxlen is not None:
            index = int(self._token_end is not None)+1
            truncate_sequences(maxlen,-index,tokens)
        return tokens

    def token_to_id(self,token):
        raise NotImplementedError

    def token_to_ids(self,tokens):
        return [self.token_to_id(token) for token in tokens]

    def encode(self,
               first_text,
               second_text=None,
               maxlen=None,
               pattern = 'S*E*E',
               truncate_from='right'):
        if is_string(first_text):
            first_tokens = self.tokenize(first_text)
        else:
            first_tokens = first_text

        if second_text is None:
            second_tokens = None
        elif is_string(second_text):
            second_text = self.tokenize(second_text)
        else:
            second_text = second_text

        if maxlen is not None:
            if truncate_from == 'right':
                index = -int(self._token_end is not None)-1
            elif truncate_from == 'left':
                index = int(self._token_end is not None)
            else:
                index = truncate_from

            if second_text is not None and pattern == 'S*E*E':
                maxlen +=1
            truncate_sequences(maxlen,index,first_tokens, second_tokens)

        first_tokens_ids = self.token_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_tokens_ids)

        if second_text is not None:
            if pattern == 'S*E*E':
                idx = int(bool(self._token_start))
                second_tokens = second_tokens[idx:]
            second_token_ids = self.token_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_tokens_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_tokens_ids,first_segment_ids

    def id_to_token(self,i):
        raise NotImplementedError

    def id_to_tokens(self,ids):
        return [self.id_to_token(i) for i in ids]

    def decode(self,ids):
        raise NotImplementedError

    def _tokenize(self,text):
        raise NotImplementedError



class Tokenizer(TokenizerBase):
    '''
    分词器
    '''
    def __init__(self,
                 token_dict,
                 do_lower_case=False,
                 word_maxlen=100,
                 init_tokens=['pad','unk','mask','start','end'],
                 **kwargs):
        super(Tokenizer, self).__init__(**kwargs)

        if is_string(token_dict):
            token_dict = load_vocab(token_dict)
        self._do_lower_case=do_lower_case
        self._word_maxlen=word_maxlen
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        self._vocab_size = len(self._token_dict)

        for token in init_tokens:
            try:
                _token_id = token_dict[getattr(self,'_token_%s' % token)]
                setattr(self,'_token_%s_id' % token,_token_id)
            except:
                pass

    def token_to_id(self,token):
        '''
        :param token:
        :return:
        '''
        return self._token_dict.get(token,self._token_unk_id)

    def id_to_token(self,i):
        '''
        :param i:
        :return:
        '''
        return self._token_dict_inv[i]

    def decode(self,ids, tokens=None):
        '''
        :param ids:
        :param tokens:
        :return:
        '''
        tokens = tokens or self.id_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]

        text,flag = '', False
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += tokens[:2]
            elif len(token) == 1 and self._is_cjk_character(token):
                text+=tokens
            elif len(token) == 1 and self._is_punctuation(token):
                text+=tokens
            elif i>0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token

        text = re.sub(' +',' ',text)
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)

        return text.strip()

    def _tokenize(self,text,pre_tokenize=True):
        '''
        :param text:
        :param pre_tokenize:
        :return:
        '''
        if self._do_lower_case:
            # if is_py2:
            #     text = unicode(text)
            text = text.lower()
            text = unicodedata.normalize('NFD',text)
            text = ''.join([
                ch for ch in text if unicodedata.category(ch) != 'Mn'
            ])

        if pre_tokenize and self._pre_tokenizer is not None:
            tokens =[]
            print("_pre_tokenizer:{}".format(self._pre_tokenizer(text)))
            for token in self._pre_tokenizer(text):
                if token in self._token_dict:
                    tokens.append(token)
                else:
                    tokens.extend((self._tokenize(text,False)))
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
            tokens.extend(self._word_piece_tokenize(word))

        return tokens

    def _word_piece_tokenize(self,word):
        if len (word) > self._word_maxlen:
            return [word]

        tokens, start, end = [], 0, 0
        while start <len(word):
            end = len(word)
            while end > start:
                sub = word[start:end]
                if start >0:
                    sub = '##' +sub
                if sub in self._token_dict:
                    break
                end -=1
            if start==end:
                return word
            else:
                tokens.append(sub)
                start = end

        return tokens


    @staticmethod
    def stem(token):
        '''
        获取token的词干
        :param token:
        :return:
        '''
        if token[:2] == '##':
            return token[2:]
        else:
            return token
    @staticmethod
    def _is_space(ch):
        '''
        空格类字符的判断
        :param ch:
        :return:
        '''
        return ch==' ' or ch =='\n' or ch =='\r' or ch == '\t' or unicodedata.category(ch)

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
