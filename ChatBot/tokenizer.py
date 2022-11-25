
from ChatBot.utils import is_string,do_lower_reg


class Tokenizer:
    def __init__(self,token_dict,pre_token=None,
                 init_tokens = ['[UNK]', '[STA]','[END]','[PAD]']):
        self.pre_token = pre_token
        if is_string(token_dict):
            token_dict = self.load_vocab(token_dict,init_tokens)
        self.token_dict = token_dict
        for token in init_tokens:
            token_low = do_lower_reg(token)
            try:
                _token_id = token_dict[token]
                setattr(self, '_token_%s_id' % token_low, _token_id)
            except:
                pass
        self.token_dict_inv = {v: k for k, v in token_dict.items()}
        self.vocab_size = len(self.token_dict)
        self.test()

    def load_vocab(self,dict_path,init_tokens, encoding='utf-8'):
        token_list = []
        token_dict = {}
        for item in init_tokens:
            token_list.append(item)
        with open(dict_path, encoding=encoding) as fr:
            for line in fr:
                token = line.split()
                token = token[0] if token else line.strip()
                token_list.append(token)
        for token in token_list:
            token_dict[token] = len(token_dict)
        return token_dict

    def test(self):
        for item in dir(self):
            if isinstance(getattr(self,item),dict):
                print(item + ":" + str(len(getattr(self, item))))

            else:
                print(item + ":" + str(getattr(self, item)))


if __name__ == '__main__':
    dict_path = './processed_data/xhj_dict.txt'
    t = Tokenizer(dict_path)
