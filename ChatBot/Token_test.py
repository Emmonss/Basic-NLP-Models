from Basic_Layer_Models.RNNs.utils.Tokenize import Tokenizer
import jieba


dict_path = './processed_data/vocab.txt'
token_path = './processed_data/token_dict.txt'
jieba.load_userdict(token_path)


def pre_token(text):
    return jieba.lcut(text)

tokenizer = Tokenizer(token_dict=dict_path,do_lower_case=True,pred_tokenizer=pre_token)



def sentence_encode(encode_text,decode_text):
    token_encoder,seg_encoder = tokenizer.encode(encode_text)
    token_decoder,seg_decoder = tokenizer.encode(decode_text)

    return token_encoder,token_decoder

def segment_decode(token_encoder,token_decoder):
    encode_text = tokenizer.id_to_tokens(token_encoder)
    decode_text = tokenizer.id_to_tokens(token_decoder)
    return encode_text,decode_text

if __name__ == '__main__':
    encode_text = "你好！"
    decode_text = "你好，傻逼！"

    token_encoder,token_decoder = sentence_encode(encode_text, decode_text)
    encode_text_1, decode_text_1 = segment_decode(token_encoder,token_decoder)

    print('init'.center(50, '-'))
    print(token_encoder)
    print(token_decoder)
    print('pre_token'.center(50, '-'))
    print(pre_token(encode_text))
    print(pre_token(decode_text))
    print('encode'.center(50, '-'))
    print(token_encoder)
    print(token_decoder)
    print('decode'.center(50, '-'))
    print(encode_text_1)
    print(decode_text_1)
