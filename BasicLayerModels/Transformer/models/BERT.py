from BasicLayerModels.Transformer.models.Transformer import Transformer
from tensorflow.keras.layers import *
from BasicLayerModels.Transformer.layers.PositionEmbedding import PositionEmbedding
from BasicLayerModels.Transformer.layers.LayerNormalization import LayerNormalization
from BasicLayerModels.Transformer.layers.MutiHeadAttention import MultiHeadAttention
from BasicLayerModels.Transformer.layers.FeedForward import FeedForward
from BasicLayerModels.Transformer.layers.BiasAdd import BiasAdd

class BERT(Transformer):
    '''
    构建BERT模型
    '''
    def __init__(self,
                 max_position,
                 segment_vocab_size=2,
                 with_pool=False,
                 with_nsp=False,
                 with_mlm=False,
                 hierarchical_position=None,
                 custom_position_ids=False,
                 shared_segment_embeddings=False,
                 **kwargs):
        super(BERT,self).__init__(**kwargs)
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.hierarchical_position = hierarchical_position
        self.custom_position_ids = custom_position_ids
        self.shared_segment_embenddings = shared_segment_embeddings
        if self.with_nsp and not self.with_pool:
            self.with_pool = True

    def get_inputs(self):
        '''
        bert 的输入是token_ids和segment_ids
        :return:
        '''
        x_in = self.apply(
            layer=Input,
            shape = (self.sequence_length,),
            name = 'Input-Token'
        )
        inputs = [x_in]

        if self.segment_vocab_size>0:
            s_in = self.apply(
                layer = Input,
                shape = (self.sequence_length,),
                name = 'Input-Segment'
            )
            inputs.append(s_in)
        if self.custom_position_ids:
            p_in = self.apply(
                layer=Input,
                shape = (self.sequence_length,),
                name = 'Input-Position'
            )
            inputs.append(p_in)

        return inputs

    def apply_embeddings(self,inputs):
        '''
        BERT的Embedding是token,position,segment三者之和
        :param inputs:
        :return:
        '''
        inputs = inputs[:]
        x = inputs.pop(0)
        if self.segment_vocab_size>0:
            s = inputs.pop(0)
        if self.custom_position_ids:
            p = inputs.pop(0)
        else:
            p=None

        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer = self.initializer,
            mask_zero = True,
            name = 'Embedding-Token'
        )

        if self.segment_vocab_size>0:
            if self.shared_segment_embenddings:
                name = 'Embedding-Token'
            else:
                name = 'Embedding-Segment'
            s = self.apply(
                inputs = s,
                layer=Embedding,
                input_dim=self.segment_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                mask_zero=True,
                name=name
            )

            #concate
            x = self.apply(
                inputs=[x,s],
                layer=Add,
                name = 'Embedding-Token-Segment'
            )

        x = self.apply(
            inputs=self.simplify([x,p]),
            layer = PositionEmbedding,
            input_dim = self.max_position,
            output_dim= self.embedding_size,
            merge_mode='add',
            hierarchical=self.hierarchical_position,
            embeddings_initializer=self.initializer,
            custom_position_ids=self.custom_position_ids,
            name='Embedding-Position'
        )

        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate = self.dropout_rate,
            name = 'Embedding-DropOut'
        )

        #当词向量维护和隐藏层的维护不等时，要通过一个全连接层
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )
        return x

    def apply_main_layers(self,inputs,index):
        '''
        BERT 的主模块 主要是self-attention 流程如下：
        Attention--Dropout--Add--LayerNorm--FeedForwardNet--Dropout--Add--LayerNorm
        :param inputs:
        :param index:
        :return:
        '''
        x = inputs
        z = self.layer_norm_conds[0]
        attention_name = 'Transformer-%d-MultiHeadSelfAttention' %index
        feed_forward_name = 'Transformer-%d-FeedForward' %index

        attention_mask = self.compute_attention_bias(index)

        xi,x,arguments = x,[x,x,x],{'a_bias':None}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.append(attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )
        return x

    def apply_final_layers(self,inputs):
        '''
        BERT最后的输出层
        :param inputs:
        :return:
        '''
        x = inputs
        z = self.layer_norm_conds[0]
        outputs = [x]

        if self.with_pool:
            x = outputs[0]
            #找到输出向量的CLS, 即张量的第一维
            x = self.apply(
                inputs=x,
                layer=Lambda,
                function=lambda x:x[:,0],
                name = 'Pooler'
            )
            pool_actication = 'tanh' if self.with_pool is True else self.with_pool
            x = self.apply(
                inputs=x,
                layer=Dense,
                units = self.hidden_size,
                activation = pool_actication,
                kernel_initializer = self.initializer,
                name = 'Pooler-Dense'
            )

        if self.with_nsp:
            #nsp 即 next-setence-prediction
            #下游任务判断句子的置信度
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=2,
                activation='softmax',
                kernel_initializer = self.initializer,
                name = 'NSP-Proba'
            )
            outputs.append(x)

        if self.with_mlm:
            #mask language models
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=Dense,
                units = self.embedding_size,
                activation = self.hidden_act,
                kernel_initializer = self.initializer,
                name = 'MLM_Dense'
            )
            x = self.apply(
                inputs=self.simplify([x,z]),
                layer=LayerNormalization,
                conditional = (z is not None),
                hidden_units = self.layer_norm_conds[1],
                hidden_activation = self.layer_norm_conds[2],
                hidden_initializer = self.initializer,
                name = 'MLM-Norm'
            )
            x = self.apply(
                inputs=x,
                layer=Embedding,
                arguments={'mode':'dense'},
                name = 'Embedding-Token'
            )
            x = self.apply(
                inputs=x,
                layer=BiasAdd,
                name = 'MLM-Bias'
            )
            mlm_activation = 'softmax' if self.with_mlm is True else self.with_mlm
            x = self.apply(
                inputs=x,
                layer=Activation,
                activation=mlm_activation,
                name = 'MLM-Activation'
            )
            outputs.append(x)

        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs)==2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        return outputs

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(BERT, self).load_variable(checkpoint, name)
        if name in [
            'bert/embeddings/word_embeddings',
            'cls/predictions/output_bias',
        ]:
            return self.load_embeddings(variable)
        elif name == 'cls/seq_relationship/output_weights':
            return variable.T
        else:
            return variable

    def create_variable(self, name, value):
        """在tensorflow中创建一个变量
        """
        if name == 'cls/seq_relationship/output_weights':
            value = value.T
        return super(BERT, self).create_variable(name, value)

    def variable_mapping(self):
        """映射到官方BERT权重格式
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
            'Embedding-Position': ['bert/embeddings/position_embeddings'],
            'Embedding-Norm': [
                'bert/embeddings/LayerNorm/beta',
                'bert/embeddings/LayerNorm/gamma',
            ],
            'Embedding-Mapping': [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ],
            'Pooler-Dense': [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ],
            'NSP-Proba': [
                'cls/seq_relationship/output_weights',
                'cls/seq_relationship/output_bias',
            ],
            'MLM-Dense': [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ],
            'MLM-Norm': [
                'cls/predictions/transform/LayerNorm/beta',
                'cls/predictions/transform/LayerNorm/gamma',
            ],
            'MLM-Bias': ['cls/predictions/output_bias'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'attention/output/LayerNorm/beta',
                    prefix + 'attention/output/LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'output/LayerNorm/beta',
                    prefix + 'output/LayerNorm/gamma',
                ],
            })

        return mapping

if __name__ == '__main__':
    from tensorflow.keras import backend
    backend.clear_session()
    bert = BERT(
        max_position=256,
        vocab_size=21128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=1248,
        hidden_act='gelu'
    )
    bert.build()
    print(bert.model.summary())

