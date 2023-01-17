import os
os.environ['TF_KERAS'] ="1"

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from BasicLayerModels.Transformer.layers.MutiHeadAttention import MultiHeadAttention
from BasicLayerModels.Transformer.layers.Concatenate1D import Concatenate1D
from tensorflow.keras import backend as K


class Transformer(object):
    '''
        模型基准类
    '''
    def __init__(self,
                 vocab_size,            #词表大小
                 hidden_size,           #编码维度
                 num_hidden_layers,     #Transformer层数
                 num_attention_heads,   #Attention的head数
                 intermediate_size,     #FeedForward的隐藏层数
                 hidden_act,            #FeedFward隐藏层激活函数
                 dropout_rate=None,     #Dropout比例
                 embedding_size=None,   #是否制定embedding_size
                 attention_head_size=None, #Attention中V的size
                 attention_key_size=None,  #Attention中Q,K的size
                 sequence_length=None,  #是否固定序列长度
                 keep_tokens=None,      #要保留词表的ID序列
                 compound_tokens=None,  #扩展Embedding
                 layers=None,           #外部传入的Keras层
                 prefix=None,           #层缀名称
                 name=None,             #模型名称
                 **kwargs
                 ):
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.layers = {} if layers is None else layers
        self.prefix = prefix or ''
        self.name = name
        self.built = False

    def build(
            self,
            attention_caches=None,
            layer_norm_cond=None,
            layer_norm_cond_hidden_size=None,
            layer_norm_cond_hidden_act=None,
            addtional_input_layers=None,
            **kwargs
    ):
        '''
        模型构建函数
        :param attention_caches:attention的K,V缓存序列字典
                            格式：
                            {
                                Attention层名： [K缓存，V缓存]
                            }
        :param layer_norm_cond: 实现“固定长度向量为条件的BERT来使用的
        :param layer_norm_cond_hidden_size:
        :param layer_norm_cond_hidden_act:
        :param addtional_input_layers:
        :param kwargs:
        :return:
        '''
        if self.built:
            return None
        inputs = self.get_inputs()
        self.set_inputs(inputs,addtional_input_layers)
        self.attention_caches = attention_caches or {}

        self.layer_norm_conds = [
            layer_norm_cond,
            layer_norm_cond_hidden_size,
            layer_norm_cond_hidden_act or 'linear'
        ]
        #Call
        ouputs = self.call(inputs)
        self.set_outputs(ouputs)
        self.model = Model(self.inputs,self.outputs,name=self.name)
        self.built = True

    def call(self,inputs):
        outputs = self.apply_embeddings(inputs)
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(outputs,i)
        outputs = self.apply_final_layers(outputs)
        return outputs

    def prefixed(self,name):
        if name is not None:
            return self.prefix + name

    def apply(self,inputs=None,layer=None,arguments=None,**kwargs):
        '''
        自动调用重名层
        :param inputs:上一层的输出
        :param layer: 调用的层类名
        :param arguments: 传递给layer的call参数
        :param kwargs: 初始化参数
        :return:
        '''
        if layer is Dropout and self.dropout_rate==0:
            return inputs
        arguments = arguments or {}
        name = self.prefixed(kwargs.get('name'))
        if name not in self.layers:
            layer = layer(**kwargs)
            name = layer.name
            self.layers[name] = layer

        if inputs is None:
            return self.layers[name]
        else:
            if isinstance(self.layers[name],MultiHeadAttention):
                if name in self.attention_caches:
                    k_cache, v_cache = self.attention_caches[name]
                    k_name, v_name = name + '-Cached-Key', name + '-Cached-Value'
                    k = Concatenate1D(name=k_name)([k_cache, inputs[1]])
                    v = Concatenate1D(name=k_name)([k_cache, inputs[2]])
                    inputs = inputs[:1] + [k,v] + inputs[3:]
            return self.layers[name](inputs,**arguments)




    ################################################
    # 实现类似于模板类的一些抽象待实现方法
    ################################################
    def get_inputs(self):
        raise NotImplementedError

    def apply_embeddings(self,inputs):
        raise NotImplementedError

    def apply_main_layers(self,inputs,index):
        raise NotImplementedError

    def apply_final_layers(self,inputs):
        raise NotImplementedError

    def compute_attention_bias(self, inputs=None):
        return self.attention_bias

    def compute_position_bias(self,inputs=None):
        return self.position_bias

    #################################################

    def set_inputs(self,inputs,addtional_input_layers=None):
        if inputs is None:
            inputs = []
        elif not isinstance(inputs,list):
            inputs = [inputs]
        inputs = inputs[:]

        if addtional_input_layers is not None:
            if not isinstance(addtional_input_layers,list):
                addtional_input_layers = [addtional_input_layers]
            inputs.extend(addtional_input_layers)

        self.inputs = inputs
        if len(inputs) > 1:
            self.input = inputs
        else:
            self.input = inputs[0]

    def set_outputs(self,outputs):
        '''
        设置output和outputs
        :param outputs:
        :return:
        '''
        if not isinstance(outputs,list):
            outputs = [outputs]

        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs)>1:
            self.output = outputs
        else:
            self.output = outputs[0]

    @property
    def initializer(self):
        '''
        默认正态分布
        :return:
        '''
        return tf.keras.initializers.TruncatedNormal(stddev=0.02)

    def simplify(self,inputs):
        '''
        过滤掉list中的None值
        :param inputs:
        :return:
        '''
        inputs = [i for i in inputs if i is not None]
        if len(inputs) == 1:
            inputs = inputs[0]

        return inputs

    def load_embeddings(self,embeddings):
        '''
        处理enbeddings
        :param embeddings:
        :return:
        '''
        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]
        if self.compound_tokens is not None:
            ext_embeddings = np.array([
                embeddings[idxs].mean(0) for idxs in self.compound_tokens
            ])
            embeddings = np.concatenate([embeddings,ext_embeddings],0)
        return embeddings

    def load_variable(self,checkpoint,name):
        '''
        加载单个变量函数
        :param checkpoint:
        :param name:
        :return:
        '''
        if isinstance(checkpoint,dict):
            return checkpoint[name]
        else:
            return tf.train.load_variable(checkpoint,name)

    def create_variable(self,name,value):
        '''
        创建一个变量
        :param name:
        :param value:
        :return:
        '''
        return K.variable(self.initializer(value.shape),name=name),value

    def variable_mapping(self):
        '''
        keras层和ckpt之间的映射表
        :return:
        '''
        return {}

    def load_weight_from_checkpoint(self,checkpoint,mapping=None):
        '''
        根据mapping从checkpoint加载权重
        :param checkpoint:
        :param mapping:
        :return:
        '''
        mapping = mapping or self.variable_mapping()
        mapping = {self.prefixed(k): v for k, v in mapping.items()}
        mapping = {k:v for k,v in mapping.items() if k in self.layers}

        weight_value_pairs = []
        for layer,variables in mapping.items():
            layer = self.layers[layer]
            weight = layer.trainable_weights
            values = [self.load_variable(checkpoint,v) for v in variables]

            if isinstance(layer,MultiHeadAttention):
                '''
                    如果key_size不等于head_size，则可以通过
                    正交矩阵将相应的权重投影到合适的shape
                '''
                count = 2
                if layer.use_bias:
                    count += 2
                heads = self.num_attention_heads
                head_size = self.attention_head_size
                key_size = self.attention_key_size
                W = np.linalg.qr(np.random.randn(key_size,head_size))[0].T
                if layer.attention_scale:
                    W = W * key_size**0.25 / head_size**0.25
                for i in range(count):
                    w,v = weight[i],values[i]
                    w_shape, v_shape = K.int_shape(w), v.shape
                    if w_shape[-1] != v_shape[-1]:
                        pre_shape = w_shape[:-1]
                        v = v.reshape(pre_shape+(heads,head_size))
                        v = np.dot(v,W)
                        v = v.reshape(pre_shape + (heads*key_size))
                        values[i] = v
            weight_value_pairs.extend(zip(weight,values))
        K.batch_set_value(weight_value_pairs)

    def save_weight_as_checkpoint(self, filename, mapping=None):
        '''
        根据mappIng将权重保存为checkpoint格式
        :param filename:
        :param mapping:
        :return:
        '''
        mapping = mapping or self.variable_mapping()
        mapping = {self.prefix(k): v for k,v in mapping.items()}
        mapping = {k:v for k,v in mapping.items() if k in self.layers}

        with tf.Graph().as_default():
            all_variables, all_values = [], []
            for layer, variables in mapping.items():
                layer = self.layers[layer]
                values = K.batch_get_value(layer.trainable_weights)
                for name, value in zip(variables,values):
                    variable,value = self.create_variable(name,value)
                    all_variables.append(variable)
                    all_values.append(value)
            with tf.Session() as sess:
                K.batch_set_value(zip(all_variables,all_values))
                saver = tf.train.Saver()
                saver.save(sess,filename)


if __name__ == '__main__':
    pass
    trans = Transformer(vocab_size=100,
                        hidden_size=256,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        intermediate_size=1,
                        hidden_act=1)
    layer = trans.apply(layer=Dense,name='Dense',units=10)

    # print(layer)