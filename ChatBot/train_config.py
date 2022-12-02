
#basic
batch_size = 2
epoch = 100
embed_dim = 100
hidden_units = 32
sent_maxlen = 30
lr = 0.001
teach_forcing_ran=0.5
pretoken_flg = True
saving_flg = True


#data_path
#1:dididi
inpath_di = './cropus/di/di_input.txt'
tarpath_di = './cropus/di/di_target.txt'
save_model_path_di = './models/di/{}'
model_name_di = 'chat_di'

#2:小黄鸡
dict_path = './processed_data/xhj_dict.txt'
inpath_xhj = './cropus/test/input.txt'
tarpath_xhj = './cropus/test/target.txt'
# inpath_xhj = './cropus/xiaohuangji/data/input.txt'
# tarpath_xhj = './cropus/xiaohuangji/data/input.txt'
save_model_path_xhj = './models/xhj/{}'
model_name_xhj = 'chat_xhj'



#dict_path
dict_vocab_pure = './processed_data/vocab_pure.txt'
params_save_name = 'param.json'

