#utf-8
import os
os.environ['TF_KERAS'] ="1"
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,Dense
from BasicLayerModels.Transformer.layers.integerize_shape import integerize_shape
import tensorflow as tf
from BasicLayerModels.Transformer.backend import sequence_masking


class MaximumEntropyMarkovModel(Layer):
    '''
    (双向)最大熵隐马尔科夫模型
    作用和用法都类似CRF 但是比CRF更好用
    '''
    def __init__(self,
                 lr_multiplier=1,
                 hidden_dim = None,**kwargs):
        super(MaximumEntropyMarkovModel,self).__init__(**kwargs)
        self.lr_multiplier = lr_multiplier
        self.hidden_dim = hidden_dim

    @integerize_shape
    def build(self, input_shape):
        super(MaximumEntropyMarkovModel, self).build(input_shape)
        output_dim = input_shape[-1]

        if self.hidden_dim is None:
            self._trans = self.add_weight(
                name='trans',
                shape = (output_dim,output_dim),
                initializer='glorot_uniform',
                trainable=True
            )
            if self.lr_multiplier !=1:
                K.set_value(
                    self._trans,
                    K.eval(self._trans)/self.lr_multiplier
                )
        else:
            self._l_trans = self.add_weight(
                name = 'l_trans',
                shape=(output_dim,self.hidden_dim),
                initializer='glorot_uniform',
                trainable=True
            )
            self._r_trans = self.add_weight(
                name='r_trans',
                shape=(output_dim,self.hidden_dim),
                initializer='glorot_uniform',
                trainable=True
            )

            if self.lr_multiplier !=1:
                K.set_value(
                    self._l_trans,
                    K.eval(self._l_trans)/self.lr_multiplier
                )
                K.set_value(
                    self._r_trans,
                    K.eval(self._r_trans)/self.lr_multiplier
                )
    @property
    def trans(self):
        if self.lr_multiplier !=1:
            return self.lr_multiplier * self._trans
        else:
            return self._trans

    @property
    def l_trans(self):
        if self.lr_multiplier !=1:
            return self.lr_multiplier * self._l_trans
        else:
            return self._l_trans

    @property
    def r_trans(self):
        if self.lr_multiplier != 1:
            return self.lr_multiplier * self._r_trans
        else:
            return self._r_trans

    def call(self,inputs,mask=None):
        if mask is not None:
            mask = K.cast(mask,K.floatx())
        return  sequence_masking(inputs,mask,1,1)

    def reverse_sequence(self,inputs,mask=None):
        if mask is None:
            return [x[:,::-1] for x in inputs]
        else:
            length = K.cast(K.sum(mask,1),'int32')
            return [tf.reverse_sequence(x,length,seq_axis=1) for x in inputs]

    def basic_loss(self,y_true,y_pred,go_backwards = False):
        '''
        ytrue是整数形式
        :param y_true:
        :param y_pred:
        :param go_backwards:
        :return:
        '''
        #导出mask并转换数据类型
        mask = K.all(K.greater(y_pred,-1e6),axis=2)
        mask = K.cast(mask,K.floatx())
        #y_true需要进一步明确type和dtype
        if self.hidden_dim is None:
            if go_backwards:#是否反转序列
                y_true, y_pred = self.reverse_sequence([y_true,y_pred],mask)
                trans = K.transpose(self.trans)
            else:
                trans = self.trans
            history = K.gather(trans,y_true)
        else:
            if go_backwards:
                y_true, y_pred = self.reverse_sequence([y_true, y_pred], mask)
                r_trans,l_trans = self.r_trans,self.l_trans
            else:
                r_trans, l_trans = self.r_trans, self.l_trans
            history = K.gather(l_trans,r_trans)
            history = tf.einsum('bnd,kd->bnk',history,r_trans)

        #计算loss
        history = K.concatenate([y_pred[:,:1]],history[:,:-1],1)
        y_pred = (y_pred+history)/2
        loss = K.sparse_categorical_crossentropy(
            y_true,y_pred,from_logits=True
        )
        return K.sum(loss*mask)/K.sum(mask)

    def sparse_loss(self, y_true, y_pred):
        """y_true需要是整数形式（非one hot）
        """
        loss = self.basic_loss(y_true, y_pred, False)
        loss = loss + self.basic_loss(y_true, y_pred, True)
        return loss / 2

    def dense_loss(self, y_true, y_pred):
        """y_true需要是one hot形式
        """
        y_true = K.argmax(y_true, 2)
        return self.sparse_loss(y_true, y_pred)

    def basic_accuracy(self, y_true, y_pred, go_backwards=False):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """
        # 导出mask并转换数据类型
        mask = K.all(K.greater(y_pred, -1e6), axis=2)
        mask = K.cast(mask, K.floatx())
        # y_true需要重新明确一下shape和dtype
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        # 反转相关
        if self.hidden_dim is None:
            if go_backwards:  # 是否反转序列
                y_true, y_pred = self.reverse_sequence([y_true, y_pred], mask)
                trans = K.transpose(self.trans)
            else:
                trans = self.trans
            histoty = K.gather(trans, y_true)
        else:
            if go_backwards:  # 是否反转序列
                y_true, y_pred = self.reverse_sequence([y_true, y_pred], mask)
                r_trans, l_trans = self.l_trans, self.r_trans
            else:
                l_trans, r_trans = self.l_trans, self.r_trans
            histoty = K.gather(l_trans, y_true)
            histoty = tf.einsum('bnd,kd->bnk', histoty, r_trans)
        # 计算逐标签accuracy
        histoty = K.concatenate([y_pred[:, :1], histoty[:, :-1]], 1)
        y_pred = (y_pred + histoty) / 2
        y_pred = K.cast(K.argmax(y_pred, 2), 'int32')
        isequal = K.cast(K.equal(y_true, y_pred), K.floatx())
        return K.sum(isequal * mask) / K.sum(mask)

    def sparse_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """
        accuracy = self.basic_accuracy(y_true, y_pred, False)
        accuracy = accuracy + self.basic_accuracy(y_true, y_pred, True)
        return accuracy / 2

    def dense_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是one hot形式
        """
        y_true = K.argmax(y_true, 2)
        return self.sparse_accuracy(y_true, y_pred)

    def get_config(self):
        config = {
            'lr_multiplier': self.lr_multiplier,
            'hidden_dim': self.hidden_dim,
        }
        base_config = super(MaximumEntropyMarkovModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
