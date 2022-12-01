
#basic
batch_size = 32
epoch = 5
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
model_name_di = 'chat_di'
#2:小黄鸡
inpath_xhj = './cropus/xiaohuangji/data/input.txt'
tarpath_xhj = './cropus/xiaohuangji/data/input.txt'
save_model_path_xhj = './models/xhj/{}'
model_name_xhj = 'chat_xhj'



#dict_path
dict_vocab_pure = './processed_data/vocab_pure.txt'
params_save_name = 'param.json'

