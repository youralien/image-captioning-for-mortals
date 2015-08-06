"""
This file contains the "bricks" for building the image-text encoder described in

Image Captioning for Mortals (Part 1):
Project 1. Rating how relevant an image and caption are to each other

Bricks are a term used in Blocks, the Theano framework, that describe parameterized
Theano operations.  You can read more about what bricks are here:
https://blocks.readthedocs.org/en/latest/bricks_overview.html
"""

from blocks.bricks import Initializable, Linear
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.recurrent import LSTM
from blocks.bricks.base import application

class Encoder(Initializable):

    def __init__(self, image_feature_dim, embedding_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.image_embedding = Linear(
              input_dim=image_feature_dim
            , output_dim=embedding_dim
            , name="image_embedding"
            )

        self.to_inputs = Linear(
              input_dim=embedding_dim
            , output_dim=embedding_dim*4 # times 4 cuz vstack(input, forget, cell, hidden)
            , name="to_inputs"
            )

        self.transition = LSTM(
            dim=embedding_dim, name="transition")

        self.children = [ self.image_embedding
                        , self.to_inputs
                        , self.transition
                        ]

    @application(
          inputs=['image_vects', 'word_vects']
        , outputs=['image_embedding', 'sentence_embedding']
        )   
    def apply(self, image_vects, word_vects):

        image_embedding = self.image_embedding.apply(image_vects)

        inputs = self.to_inputs.apply(word_vects)
        
        # shuffle dimensions to correspond to (sequence, batch, features)
        inputs = inputs.dimshuffle(1, 0, 2)
        
        hidden, cells = self.transition.apply(inputs=inputs, mask=None)

        # last hidden state represents the accumulation of word embeddings 
        # (i.e. the sentence embedding)
        sentence_embedding = hidden[-1]

        return image_embedding, sentence_embedding

if __name__ == '__main__':
    import numpy as np
    import theano
    from theano import tensor

    def test_encoder():
        image_vects = tensor.matrix('image_vects')
        word_vects = tensor.tensor3('word_vects')
        batch_size = 2
        image_feature_dim = 64
        seq_len = 4
        embedding_dim = 300


        s = Encoder(
                  image_feature_dim=image_feature_dim
                , embedding_dim=embedding_dim
                , biases_init=Constant(0.)
                , weights_init=IsotropicGaussian(0.02)
                )
        s.initialize()
        iem, sem = s.apply(image_vects, word_vects)

        image_vects_tv = np.zeros((batch_size, image_feature_dim), dtype='float32')
        word_vects_tv = np.zeros((batch_size, seq_len, embedding_dim), dtype='float32')

        # expecting sentence embedding to be [batch_size, embedding_dim]
        f = theano.function([image_vects, word_vects], [iem, sem])
        i_emb, s_emb = f(image_vects_tv, word_vects_tv)

        print("""
            batch_size: %d
            image_feature_dim: %d
            sequence length: %d
            embedding dim: %d \n"""
            % (
                batch_size
              , image_feature_dim
              , seq_len
              , embedding_dim)
        )

        print "input image vectors: ", (batch_size, image_feature_dim)
        print "input word vectors: ", (batch_size, seq_len, embedding_dim)
        print "image embedding: ", i_emb.shape
        print "sentence embedding: ", s_emb.shape

    test_encoder()