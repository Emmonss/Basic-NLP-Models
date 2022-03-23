import os
os.environ['TF_KERAS'] ="1"
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Embedding as tf_embd
class Embedding(tf_embd):
    '''
        拓展keras自带的embedding层
    '''
    def compute_mask(self, inputs, mask=None):
        '''
        for T5 保证第一个token不被mask
        :param inputs:
        :param mask:
        :return:
        '''
        if K.ndim(inputs)==2:
            mask = super(Embedding, self).compute_mask(inputs,mask)
            if mask is not None:
                mask1 = K.ones_like(mask[:,:1],dtype='bool')
                mask2 = mask[:,1:]
                return K.concatenate([mask1,mask2],1)
            else:
                return mask

    def call(self,inputs,mode = 'embedding'):
        '''
        新增mode参数，可以为embedding或dense,如果是embed，则为普通的embed层。如果为dense,则等价于无bias的层
        :param inputs:
        :param mode:
        :return:
        '''
        if mode =='embedding':
            return super(Embedding, self).call(inputs)
        else:
            kernel = K.transpose(self.embeddings)
            return K.dot(inputs,kernel)

    def compute_output_shape(self, input_shape):
        '''
        因为缓存的mode是不准的，所以要重新计算一下
        :param input_shape:
        :return:
        '''
        if len(input_shape) == 2:
            return super(Embedding,self).compute_output_shape(input_shape)
        else:
            return input_shape[:2]+K.int_shape(self.embeddings)[0]