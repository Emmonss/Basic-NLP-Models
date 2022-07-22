
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Embedding,Bidirectional,LSTM
import tensorflow as tf

from Basic_Layer_Models.RNNs.models.NLU_Basic import NLUModel
from Basic_Layer_Models.RNNs.layers import SeqAttention,SeqDcoder,SeqEncoder

class Seq2SeqAttention(NLUModel):
    def __init__(self,
                 vocab_input_size,
                 vocab_output_size,
                 embeding_dim,
                 max_sent_len,
                 units,
                 batch_size
                 ):
        self.vocab_input_size = vocab_input_size
        self.vocab_output_size = vocab_output_size
        self.embeding_dim = embeding_dim
        self.max_sent_len = max_sent_len
        self.units = units
        self.batch_size = batch_size

        super(Seq2SeqAttention, self).__init__()

    def build_model(self):
        inputs = Input(shape=(self.max_sent_len,), name='inputs')
    def compile_model(self):
        pass

if __name__ == '__main__':
    pass
