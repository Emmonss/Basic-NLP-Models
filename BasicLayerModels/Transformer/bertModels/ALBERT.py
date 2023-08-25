from BasicLayerModels.Transformer.bertModels.BERT import BERT
from BasicLayerModels.Transformer.layers.MutiHeadAttention import MultiHeadAttention
from BasicLayerModels.Transformer.layers.LayerNormalization import LayerNormalization
from BasicLayerModels.Transformer.layers.FeedForward import FeedForward
from tensorflow.keras.layers import Add,Dropout

class ALBERT(BERT):
    '''ALBERT模型
    '''
    def apply_main_layers(self,inputs,index):
        '''
        ALBERT基于self-Attention的模块
        Att--Add--LN--FFN--Add--LN
        '''
        x=inputs
        z=self.layer_norm_conds[0]

        attention_name = 'Transformer-MultiHeadSelfAttention'
        feed_forward_name = 'Transformer-FeedForward'
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
            inputs = [xi,x],
            layer=Add,
            name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs = self.simplify([x,z]),
            layer = LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

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

    def variable_mapping(self):
        '''映射到官方ALBERT权重格式
        '''
        mapping = super(ALBERT, self).variable_mapping()

        prefix = 'bert/encoder/transformer/group_0/inner_group_0/'
        mapping.update({
            'Transformer-MultiHeadSelfAttention': [
                prefix + 'attention_1/self/query/kernel',
                prefix + 'attention_1/self/query/bias',
                prefix + 'attention_1/self/key/kernel',
                prefix + 'attention_1/self/key/bias',
                prefix + 'attention_1/self/value/kernel',
                prefix + 'attention_1/self/value/bias',
                prefix + 'attention_1/output/dense/kernel',
                prefix + 'attention_1/output/dense/bias',
            ],
            'Transformer-MultiHeadSelfAttention-Norm': [
                prefix + 'LayerNorm/beta',
                prefix + 'LayerNorm/gamma',
            ],
            'Transformer-FeedForward': [
                prefix + 'ffn_1/intermediate/dense/kernel',
                prefix + 'ffn_1/intermediate/dense/bias',
                prefix + 'ffn_1/intermediate/output/dense/kernel',
                prefix + 'ffn_1/intermediate/output/dense/bias',
            ],
            'Transformer-FeedForward-Norm': [
                prefix + 'LayerNorm_1/beta',
                prefix + 'LayerNorm_1/gamma',
            ],
        })

        return mapping

if __name__ == '__main__':
    pass