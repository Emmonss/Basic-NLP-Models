
import numpy as np
import tensorflow as tf
from BasicLayerModels.Transformer.backend.snippets import is_string

class DataGenerator(object):
    '''
    数据生成器模板
    '''
    def __init__(self,data,batch_size=32,buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data,'__len__'):
            self.steps = len(self.data)//self.batch_size
            if len(self.data)%self.batch_size != 0:
                self.steps+=1
        else:
            self.steps = None

        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self,random=False):
        '''
        采样函数
        :param random:
        :return:
        '''
        if random:
            if self.steps is None:
                def generator():
                    caches,isfull = [],False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull= True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)
            else:
                def generator():
                    for i in np.random.permutation(len(self.data)):
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False,d_current
            d_current = d_next
        yield True,d_current

    def __iter__(self,random=False):
        raise NotImplementedError

    def forfit(self,random=True):
        while True:
            for d in self.__iter__(random):
                yield d

    def to_dataset(self,types,shapes,names=None,padded_batch=False):
        '''
        转为tf.data.Dataset格式
        如果传入names的话，自动把数据包装成dict形式
        :param types:
        :param shapes:
        :param names:
        :param padded_batch:
        :return:
        '''
        if names is None:
            generator = self.forfit
        else:
            if is_string(names):
                warps = lambda k,v:{k:v}
            elif is_string(names[0]):
                warps = lambda k,v:dict(zip(k,v))
            else:
                warps = lambda k,v: tuple(
                    dict(zip(i,j)) for i,j in zip(k,v)
                )
            def generator():
                for d in self.forfit():
                    yield warps(names,d)
            types = warps(names,types)
            shapes = warps(names,shapes)

        if padded_batch:
            dataset = tf.data.Dataset.from_generator(
                generator,output_types=types
            )
            dataset = dataset.padded_batch(self.batch_size,shapes)
        else:
            dataset = tf.data.Dataset.from_generator(
                generator,output_types=types,output_shapes=shapes
            )
            dataset  = dataset.batch(self.batch_size)
        return dataset