#utf-8

import os,sys,six,re,json
import logging
import numpy as np
from collections import defaultdict
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras

_open_ = open
is_py2 = six.PY2

if not is_py2:
    basestring = str


def to_array(*args):
    '''
    批量转numpy的array
    :param args:
    :return:
    '''
    results = [np.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    return results

def is_string(s):
    '''
    判断是否是字符串
    :param s:
    :return:
    '''
    return isinstance(s,basestring)

def strQ2B(ustring):
    '''
    全符号转半角符号
    :param ustring:
    :return:
    '''
    rstring = ''
    for uchar in ustring:
        insize_code = ord(uchar)
        #全角空格直接转换
        if insize_code == 12288:
            insize_code=32
        elif(insize_code>=65281 and insize_code <=65374):
            insize_code -=65248
        rstring+=chr(insize_code)
    return rstring

def string_matching(s,keywords):
    '''
    s 是否至少包含keywords中的一个
    :param s:
    :param keywords:
    :return:
    '''
    if not isinstance(keywords,list):
        keywords = [keywords]
    for item in keywords:
        if re.search(item,s):
            return True
    return False

def convert_to_unicode(text,encoding='utf-8',errors='ignore'):
    '''
    字符串转换为unicode格式
    :param text:
    :param encoding:
    :param errors:
    :return:
    '''
    if is_py2:
        if isinstance(text,str):
            text = text.decode(encoding,errors=errors)
    else:
        if isinstance(text,bytes):
            text = text.decode(encoding,errors=errors)
    return text


def convert_to_str(text, encoding='utf-8',errors='ignore'):
    '''
    字符串转换为str格式
    :param text:
    :param encoding:
    :param errors:
    :return:
    '''
    # if is_py2:
    #     if isinstance(text,unicode):
    #         text = text.encode(encoding,errors=errors)
    # else:
    if isinstance(text,bytes):
        text = text.decode(encoding,errors=errors)
    return text


class open:
    '''
        增加open的索引功能 兼容py2 py3
    '''
    def __init__(self,name,mode='r',encoding=None,errors='strict',indexable=False):
        self.name = name
        if is_py2:
            self.file = _open_(name,mode)
        else:
            self.file = _open_(name,mode,encoding=encoding,errors=errors)
        self.encoding = encoding
        self.errors = errors
        self.iterator = None
        if indexable:
            if is_string(indexable) and os.path.exists(indexable):
                self.offsets = json.load(_open_(indexable))
            else:
                self.create_indexes()
                if is_string(indexable):
                    json.dump(self.offsets,_open_(indexable,'w'))

    def create_indexes(self):
        print("create indexing ....")
        self.offsets,offset = [],0
        pbar = keras.utils.Progbar(os.path.getsize(self.name))
        while self.readline():
            self.offsets.append(offset)
            offset = self.tell()
            pbar.update(offset)
        self.seek(0)
        print('indexes created.')

    def __getitem__(self, key):
        self.seek(self.offsets[key])
        l = self.readline()
        if self.encoding:
            l = convert_to_unicode(l,self.encode, self.errors)
        return l

    def __len__(self):
        return len(self.offsets)

    def __iter__(self):
        if hasattr(self,'offsets'):
            for i in range(len(self)):
                yield self[i]
        else:
            for l in self.file:
                if self.encoding:
                    l = convert_to_unicode(l,self.encoding,self.errors)
                yield l

    def next(self):
        if self.iterator is None:
            self.iterator = self.__iter__()
        return next(self.iterator)

    def __next__(self):
        return self.next()

    def read(self):
        text = self.file.read()
        if self.encoding:
            text = convert_to_unicode(text,self.encoding,self.errors)
        return text

    def readline(self):
        text = self.file.readline()
        if self.encoding:
            text = convert_to_unicode(text,self.encoding, self.errors)
        return text

    def readlines(self):
        if self.encoding:
            return [
                convert_to_unicode(text,self.encoding,self.errors)
                for text in self.file.readlines()
            ]
        return self.file.readlines()

    def write(self,text):
        if self.encoding:
            text = convert_to_str(text,self.encoding,self.errors)
        self.file.write(text)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

    def tell(self):
        return self.file.tell()

    def seek(self,offset=0):
        return self.file.seek(offset)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()


def parallel_apply(
        func,
        iterable,
        workers,
        max_queue_size,
        callback=None,
        dummy=False,
        random_seeds=True
):
    '''
    多进程或多线程的将func应用到每个iterable里
    :param func:
    :param iterable:
    :param workers:
    :param max_queue_size:
    :param callback:
    :param dummy:
    :param random_seeds:
    :return:
    '''
    if dummy:
        from multiprocessing.dummy import Pool, Queue
    else:
        from multiprocessing import  Pool,Queue

    in_queue, out_queue, seed_queue = Queue(max_queue_size), Queue(), Queue()
    if random_seeds is True:
        random_seeds = [None]*workers
    elif random_seeds is None or random_seeds is False:
        random_seeds = []
    for seed in random_seeds:
        seed_queue.put(seed)

    def worker_step(in_queue, out_queue):
        '''
        单步函数包装成循环执行l
        :param in_queue:
        :param out_queue:
        :return:
        '''
        if not seed_queue.empty():
            np.random.seed(seed_queue.get())
        while True:
            i,d = in_queue.get()
            r = func(d)
            out_queue.put((i,r))

    pool = Pool(workers, worker_step,(in_queue,out_queue))

    if callback is None:
        results = []

    #后期处理函数
    def process_out_queue():
        out_count = 0
        for _ in range(out_queue.qsize()):
            i,d = out_queue.get()
            out_count+=1
            if callback is None:
                results.append((i,d))
            else:
                callback(d)
        return out_count

    # 存入数据，取出结果
    in_count, out_count = 0,0
    for i,d in enumerate(iterable):
        in_count+=1
        while True:
            try:
                in_queue.put((i,d),block=False)
                break
            except six.moves.queue.Full:
                out_count+=process_out_queue()
        if in_count % max_queue_size == 0:
            out_count+=process_out_queue()

    while out_count != in_count:
        out_count +=process_out_queue()

    pool.terminate()


    if callback is None:
        results = sorted(results,key=lambda r:r[0])
        return [r[1] for r in results]

def sequence_padding(inputs,length=None,value=0,seq_dims=1,mode='post'):
    '''
    Numpy函数，将序列padding到固定长度
    :param inputs:
    :param length:
    :param value:
    :param seq_dim:
    :param mode:
    :return:
    '''
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs],axis=0)
    elif not hasattr(length,'__getitem__'):
        length = [length]

    slice = [np.s_[:length[i]] for i in range(seq_dims)]
    slice = tuple(slice) if len(slice)>1 else slice[0]
    pad_with = [(0,0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slice]
        for i in range(seq_dims):
            if mode == 'post':
                pad_with[i] = (0,length[i] - np.shape(x)[i])
            elif mode =='pre':
                pad_with[i] = (length[i] - np.shape(x)[i],1)
            else:
                raise ValueError("mode argument must be post or pre")
        x = np.pad(x,pad_with,'constant',constant_values=value)
        outputs.append(x)
    return np.array(outputs)

def truncate_sequences(max_len,indices,*sequences):
    '''
    截断总长度至不超过maxlen
    :param max_len:
    :param indices:
    :param sequences:
    :return:
    '''
    sequences = [s for s in sequences if s]
    if not isinstance(indices,(list,tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) >max_len:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences

def text_segmentate(text,maxlen,seps='\n',strips=None):
    '''
    将文本按照标签符号划分为若干个短句
    :param text:
    :param maxlen:
    :param seps:
    :param strips:
    :return:
    '''
    text = text.strip().strtp(strips)
    if seps and len(text)> maxlen:
        pieces = text.split(seps[0])
        text,texts = '',[]
        for i,p in enumerate(pieces):
            if text and p and len(text)+len(p) > maxlen-1:
                texts.extend(text_segmentate(text,maxlen,seps[1:],strips))
                text = ''
            if i+1 == len(pieces):
                text = text+p
            else:
                text = text+p+seps[0]
        if text:
            texts.extend(text_segmentate(text,maxlen,seps[1:],strips))
        return texts
    else:
        return [text]



def longest_common_substring(source,target):
    '''
    最长公共子串  source,target 的最长公共切片区域
    :param source:
    :param target:
    :return:
    '''
    c,l,span = defaultdict(int),0,(0,0,0,0)
    for i,si in enumerate(source,1):
        for j,tj in enumerate(target,1):
            if si == tj:
                c[i,j] = c[i-1,j-1]+1
                if c[i,j]>l:
                    l = c[i,j]
                    span = (i-1,i,j-1,j)
    return l,span


if __name__ == '__main__':
    pass
    # x = [
    #     [1,2,3,4,5],
    #     [1,2,3]
    # ]
    # x = np.array(x)
    # print(x)
    # o = sequence_padding(x,length=10)
    # print(o)

