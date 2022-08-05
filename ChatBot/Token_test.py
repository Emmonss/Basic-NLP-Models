from Basic_Layer_Models.RNNs.utils.Tokenize import Tokenizer

dict_path = './processed_data/vocab.txt'

tokenizer = Tokenizer(token_dict=dict_path,do_lower_case=True)



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
    print('encode'.center(50, '-'))
    print(token_encoder)
    print(token_decoder)
    print('decode'.center(50, '-'))
    print(encode_text_1)
    print(decode_text_1)
