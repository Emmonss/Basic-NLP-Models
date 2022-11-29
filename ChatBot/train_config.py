
#basic
batch_size = 32
epoch = 2
embed_dim = 256
hidden_units = 256
sent_maxlen = 50
lr = 0.0001
teach_forcing_ran=0.1
pretoken_flg = True
saving_flg = True


#data_path
#1:dididi
inpath_di = './cropus/di/di_input.txt'
tarpath_di = './cropus/di/di_target.txt'
save_model_path_di = './models/di/{}'
model_name = 'chat_di'


#dict_path
dict_vocab_pure = './processed_data/vocab_pure.txt'

