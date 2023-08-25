
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda
class LM_MASK(object):
    '''下三角AttentionMask
    '''
    def compute_attention_bias(self,inputs=None):
        '''通过idxs序列
        :param inputs:
        :return:
        '''
        if self.attention_bias is None:
            def lm_mask(s):
                seq_len=K.shape(s)[1]
                idxs = K.arange(0,seq_len)
                mask = idxs[None,:] <= idxs[:,None]
                mask = K.cast(mask,K.floatx())
                return -(1-mask[None,None])*1e12

            self.attention_bias=self.apply(
                inputs = self.inputs[0],
                layer = Lambda,
                function=lm_mask,
                name='Attention-LM-Mask'
            )
        return self.attention_bias

class UniLM_Mask(object):
    '''UniLM:Seq2Seq模型用
    '''
    def compute_attention_bias(self,inputs=None):
        '''通过idxs序列
        :param inputs:
        :return:
        '''
        if self.attention_bias is None:
            def unilm_mask(s):
                idxs = K.cumsum(s,axis=1)
                mask = idxs[:,None,:] <= idxs[:,:,None]
                mask = K.cast(mask,K.floatx())
                return -(1-mask[None,None])*1e12

            self.attention_bias=self.apply(
                inputs = self.inputs[0],
                layer = Lambda,
                function=unilm_mask,
                name='Attention-UniLM-Mask'
            )
        return self.attention_bias