
import os,sys
from distutils.util import strtobool
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest,tf_inspect
from tensorflow.python.eager import tape
from tensorflow.python.ops.custom_gradient import _graph_mode_decorator

is_tf_keras = strtobool(os.environ.get('TF_KERAS','0'))

if is_tf_keras:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    sys.modules['keras'] = keras
else:
    import keras
    import keras.backend as K

#是否启用重计算（通过时间换空间）
do_recompute = strtobool(os.environ.get('RECOMPUTE','0'))

def gelu_erf(x):
    '''
        gelu_erf损失函数计算
    :param x:
    :return:
    '''
    return 0.5 * x * (1.0 + tf.math.erf(x/np.sqrt(2.0)))

def gelu_tanh(x):
    '''
        基于tanh函数的gelu
    :param x:
    :return:
    '''
    cdf = 0.5 * (1.0*K.tanh(np.sqrt(2/np.pi)*(x+0.044715*K.pow(x,3))))
    return x * cdf

def set_gelu(version):
    '''
        gelu版本
    :param version:
    :return:
    '''
    version = version.lower()
    assert version in ['erf','tanh'],'gelu version must be erf or tanh'
    if version == 'erf':
        keras.utils.get_custom_objects()['gelu'] = gelu_erf
    else:
        keras.utils.get_custom_objects()['gelu'] = gelu_tanh

def piecewise_linear(t,shedule):
    '''
    分段线性函数
    :param t:下标
    :param shedule: 类似{1000:1,2000:0.1}的字典，表示t属于[0:1000]时，输出从0均匀增加至1，
                    [1000,2000]时 输出均匀从1降至0.1 [2000,正无穷]时保持不变
    :return:
    '''
    shedule = sorted(shedule.items())
    if shedule[0][0]!=0:
        shedule = [(0,0.0)]+shedule

    x = K.constant(shedule[0][1],dtype=K.floatx())
    t = K.cast(t,K.floatx())
    for i in range(len(shedule)):
        t_begin = shedule[i][0]
        x_begin = x
        if i != len(shedule)-1:
            dx = shedule[i+1][1] - shedule[i][1]
            dt = shedule[i+1][0] - shedule[i][0]
            slope = 1.0*dx/dt
            x = shedule[i][1] + slope * (t - t_begin)
        else:
            x = K.constant(shedule[i][1],dtype=K.floatx())
        x = K.switch(t>=t_begin,x,x_begin)
    return x

def search_layer(inputs,name,exclude_from=None):
    '''
    根据inputs和name来搜索层，根据inputs一直往上递归搜索，直到找到目标层name为止，找不到就拉倒 返回None
    :param inputs: 某个层的输出
    :param name: 目标层的名字
    :param exclude_from:
    :return:
    '''
    if exclude_from is None:
        exclude_from = set()
    if isinstance(inputs,keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]
    if layer.name == name:
        return layer
    elif layer in exclude_from:
        return None
    else:
        exclude_from.add(layer)
        if isinstance(layer,keras.models.Model):
            model = layer
            for layer in model.layers:
                if layer.name == name:
                    return layer
        inbound_layers = layer._inbound_nodes[0]._inbound_layers

        if not isinstance(inbound_layers,list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers)>0:
            for layer in inbound_layers:
                layer = search_layer(layer,name,exclude_from)
                if layer is not None:
                    return layer

def sequence_masking(x,mask,mode=0,axis=None):
    '''
    为序列条件mask的函数
    :param x:
    :param mask:
    :param mode:
    :param axis:
    :return:
    '''
    if mask is None or mode not in [0,1]:
        return x
    else:
        if axis is None:
            axis=1
        if axis ==-1:
            axis = K.ndim(x)-1
        assert axis >0,"axis must be greater than 0"
        for _ in range(axis-1):
            mask = K.expand_dims(mask,1)
        for _ in range(K.ndim(x) - K.ndim(mask) - axis +1):
            mask = K.expand_dims(mask,K.ndim(mask))
        if mode == 0:
            return x*mask
        else:
            return x - (1-mask) *1e12


def batch_gather(params,indices):
    '''

    :param params:
    :param indices:
    :return:
    '''
    if K.dtype(indices)[:3] !='int':
        indices = K.cast(indices,'int32')

    try:
        return tf.gather(params,indices,batch_dims=K.ndim(indices)-1)
    except Exception as e1:
        try:
            return tf.batch_gather(params,indices)
        except Exception as e2:
            raise ValueError("%s\n%s\n"%(e1.message,e2.message))


def pool1d(x,pool_size,strides=1,padding='valid',data_format = None,pool_mode = max):
    '''
    池化函数
    :param x:
    :param pool_size:
    :param strides:
    :param padding:
    :param data_format:
    :param pool_mode:
    :return:
    '''
    x = K.expand_dims(1,pool_size)
    x = K.pool2d(x,
                 pool_size=(1,pool_size),
                 strides=(1,strides),
                 padding=padding,
                 data_format=data_format,
                 pool_mode = pool_mode)
    return x[:,0]

def dicisible_temporal_padding(x,n):
    '''
    将一维向量padding到能被n整除
    :param x:
    :param n:
    :return:
    '''
    r_len = K.shape(x)[1] % n
    p_len = K.switch(r_len>0,n-r_len,0)
    return K.temporal_padding(x,(0,p_len))

def swish(x):
    '''
    swidh函数
    :param x:
    :return:
    '''
    return tf.nn.swish(x)

def leaky_relu(x,alpha=0.2):
    return tf.nn.leaky_relu(x,alpha=alpha)

class Sinusodal(keras.initializers.Initializer):
    '''
        sin-cos PE位置向量生成器
    '''
    def __call__(self, shape, dtype):
        vocab_size,depth = shape
        embeddings = np.zeros(shape)
        for pos in range(vocab_size):
            for i in range(depth//2):
                theta = pos/np.pow(10000,2. *i/depth)
                embeddings[pos,2*i] = np.sin(theta)
                embeddings[pos,2*i+1] = np.cos(theta)
        return embeddings

def sysbolic(f):
    '''
    恒等装饰器
    :param f:
    :return:
    '''
    return f

def graph_mode_decorator(f,*args,**kwargs):
    if tf.__version__ <'2.1':
        return _graph_mode_decorator(f,*args,**kwargs)
    else:
        return _graph_mode_decorator(f, args, kwargs)

def recompute_grad(call):
    '''
    重计算装饰器
    :param call:
    :return:
    '''

    if not do_recompute:
        return call
    def inner(self,inputs,**kwargs):
        '''
        定义需要求梯度的函数以及重新定义求梯度的过程
        :param self:
        :param inputs:
        :param kwargs:
        :return:
        '''
        flag_inputs = nest.flatten(inputs)
        call_args = tf_inspect.getfullargspec(call).args
        for key in ['mask','training']:
            if key not in call_args and key in kwargs:
                del kwargs[key]

        def kernel_call():
            '''
            定义forword计算
            :return:
            '''
            return call(self,inputs,**kwargs)

        def call_and_grad(*inputs):
            '''
            定义forward and backward计算
            :param inputs:
            :return:
            '''
            if is_tf_keras:
                with tape.stop_recording():
                    outputs = kernel_call()
                    outputs = tf.identity(outputs)
            else:
                outputs = kernel_call()

            def grad_fn(doutputs,vatiables = None):
                wathces = list(inputs)
                if vatiables is not None:
                    wathces+=list(vatiables)
                with tf.GradientTape() as t:
                    t.watch(wathces)
                    with tf.control_dependencies([doutputs]):
                        outputs = kernel_call()
                grads = t.gradient(
                    outputs,wathces,output_gradients=[doutputs]
                )
                del t
                return grads[:len(inputs)],grads[len(inputs):]
            return outputs,grad_fn

        if is_tf_keras:
            outputs,grad_fn = call_and_grad(inputs)
            flat_outputs = nest.flatten(outputs)

            def actual_grad_fn(*doutputs):
                grads = grad_fn(*doutputs,vatiables=self.trainable_weight)
                return grads[0] + grads[1]

            watches = flag_inputs + self.trainable_weight
            watches = [tf.convert_to_tensor(x) for x in watches]
            tape.record_operation(
                call.__name__,flat_outputs,watches,actual_grad_fn
            )
            return outputs
        else:
            return graph_mode_decorator(call_and_grad,*flag_inputs)
    return inner

K.symbolic = getattr(K,'symbolic',None) or sysbolic

custom_objects = {
    'gelu_erf':gelu_erf,
    'gelu_tanh':gelu_tanh,
    'gelu':gelu_erf,
    'swish':swish,
    'leaky_relu':leaky_relu,
    'Sinusoidal':Sinusodal
}

keras.utils.get_custom_objects().update(custom_objects)