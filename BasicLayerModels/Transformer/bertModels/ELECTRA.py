from BasicLayerModels.Transformer.bertModels import BERT
from BasicLayerModels.Transformer.layers.MutiHeadAttention import MultiHeadAttention
from BasicLayerModels.Transformer.layers.LayerNormalization import LayerNormalization
from BasicLayerModels.Transformer.layers.FeedForward import FeedForward
from tensorflow.keras.layers import *

class ELECTRA(BERT):
    def __init__(self,max_position,**kwargs):
        super(ELECTRA,self).__init__(max_position,**kwargs)

    def apply_final_layers(self,inputs):
        x = inputs

        if self.with_discriminator:
            if self.with_discriminator is True:
                final_activation = 'sigmod'
            else:
                final_activation = self.with_discriminator
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name='Discriminator-Dense'
            )
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=1,
                activation=final_activation,
                kernel_initializer=self.initializer,
                name='Discriminator-Prediction'
            )
        return x

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(ELECTRA, self).load_variable(checkpoint, name)
        if name == 'electra/embeddings/word_embeddings':
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self):
        mapping = super(ELECTRA, self).variable_mapping()
        mapping['Embedding-Mapping'] = [
            'electra/embeddings_project/kernel',
            'electra/embeddings_project/bias',
        ]
        mapping = {
            k: [i.replace('bert/', 'electra/') for i in v]
            for k, v in mapping.items()
        }
        mapping['Discriminator-Dense'] = [
            'discriminator_predictions/dense/kernel',
            'discriminator_predictions/dense/bias',
        ]
        mapping['Discriminator-Prediction'] = [
            'discriminator_predictions/dense_1/kernel',
            'discriminator_predictions/dense_1/bias',
        ]
        return mapping
