# scientific python
import theano
from theano import tensor

# blocks model building
from blocks.initialization import Uniform, Constant
from blocks.graph import ComputationGraph

# blocks model training
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, Adam
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import Printing

# cuboid
from cuboid.extensions import UserFunc, LogToFile

# local imports
from bricks import Encoder
from pipeline import DataETL, ModelEval
from dataset import coco, sbu
from utils import l2norm, ModelIO
from config import MODEL_FILES_DIR

def trainencoder(
      sources = ("image_vects", "word_vects")
    , sources_k = ("image_vects_k", "word_vects_k")
    , batch_size=128
    , embedding_dim=300
    , n_captions=5
    , n_sbu=None
    , separate_emb=False
    , test_size=1000 # per dataset
    , mode='dev'
    ):
    trX, teX, trY, teY = coco(mode=mode, n_captions=n_captions, test_size=test_size)
    if n_sbu:
        sbutrX, sbuteX, sbutrY, sbuteY = sbu(mode=mode, test_size=test_size)
        pairs = (
              (trX, sbutrX)
            , (teX, sbuteX)
            , (trY, sbutrY)
            , (teY, sbuteY)
            )

        for coco_data, sbu_data in pairs:
            if isinstance(coco_data, list):
                coco_data.extend(sbu_data)                        

    print("n_train: %d" % len(trX))
    print("n_test: %d" % len(teX))

    # # # # # # # # # # #
    # Modeling Building #
    # # # # # # # # # # #

    s = Encoder(
          image_feature_dim=4096
        , embedding_dim=embedding_dim
        , biases_init=Constant(0.)
        , weights_init=Uniform(width=0.08)
        )
    s.initialize()

    image_vects = tensor.matrix(sources[0]) # named to match the source name
    word_vects = tensor.tensor3(sources[1]) # named to match the source name
    image_vects_k = tensor.matrix(sources_k[0]) # named to match the contrastive source name
    word_vects_k = tensor.tensor3(sources_k[1]) # named to match the contrastive source name

    # image_vects.tag.test_value = np.zeros((2, 4096), dtype='float32')
    # word_vects.tag.test_value = np.zeros((2, 15, 50), dtype='float32')
    # image_vects_k.tag.test_value = np.zeros((2, 4096), dtype='float32')
    # word_vects_k.tag.test_value = np.zeros((2, 15, 50), dtype='float32')

    # learned image embedding, learned sentence embedding
    lim, ls = s.apply(image_vects, word_vects)

    # learned constrastive im embedding, learned contrastive s embedding
    lcim, lcs = s.apply(image_vects_k, word_vects_k)

    # l2norms
    lim = l2norm(lim)
    lcim = l2norm(lcim)
    ls = l2norm(ls)
    lcs = l2norm(lcs)

    margin = 0.2 # alpha term, should not be more than 1!

    # pairwise ranking loss (https://github.com/youralien/skip-thoughts/blob/master/eval_rank.py)
    cost_im = margin - (lim * ls).sum(axis=1) + (lim * lcs).sum(axis=1)
    cost_im = cost_im * (cost_im > 0.) # this is like the max(0, pairwise-ranking-loss)
    cost_im = cost_im.sum(0)

    cost_s = margin - (ls * lim).sum(axis=1) + (ls * lcim).sum(axis=1)
    cost_s = cost_s * (cost_s > 0.) # this is like max(0, pairwise-ranking-loss)
    cost_s = cost_s.sum(0)

    cost = cost_im + cost_s
    cost.name = "pairwise_ranking_loss"

    # function(s) to produce embedding
    if separate_emb:
        img_encoder = theano.function([image_vects], lim)
        txt_encoder = theano.function([word_vects], ls)
    f_emb = theano.function([image_vects, word_vects], [lim, ls])

    if n_sbu:
        sbuname = "sbu%d+" % n_sbu
    else:
        sbuname = ''
    name = "%sTEST" % (sbuname)
    savename = MODEL_FILES_DIR + name

    def save_function(self):
        if separate_emb:
            ModelIO.save(
                  img_encoder
                , savename + "_Img")
            ModelIO.save(
                  txt_encoder
                , savename + "_Txt")
        ModelIO.save(f_emb, savename)
        print "Similarity Embedding function(s) saved while training"

    def rank_function(stream):
        images, captions, _0, _1 = stream.get_epoch_iterator().next()
        image_embs, caption_embs = f_emb(images, captions)
        ModelEval.ImageSentenceRanking(image_embs, caption_embs)

    def rank_coco(self=None):
        # Get 1000 images / captions to test rank
        stream = DataETL.getFinalStream(teX, teY, sources=sources,
                            sources_k=sources_k, batch_size=test_size,
                            shuffle=True)
        print "COCO test"
        rank_function(stream)

    def rank_sbu(self=None):
        stream = DataETL.getFinalStream(sbuteX, sbuteY, sources=sources,
                            sources_k=sources_k, batch_size=test_size,
                            shuffle=True)
        print "SBU test"
        rank_function(stream)

    def rank_em(self=None):
        rank_coco()
        if n_sbu:
            rank_sbu()

    cg = ComputationGraph(cost)

    # # # # # # # # # # #
    # Modeling Training #
    # # # # # # # # # # #

    algorithm = GradientDescent(
          cost=cost
        , parameters=cg.parameters
        , step_rule=Adam(learning_rate=0.0002)
        )
    main_loop = MainLoop(
          model=Model(cost)
        , data_stream=DataETL.getFinalStream(trX, trY, sources=sources,
              sources_k=sources_k, batch_size=batch_size)
        , algorithm=algorithm
        , extensions=[
              DataStreamMonitoring(
                  [cost]
                , DataETL.getFinalStream(trX, trY, sources=sources,
                      sources_k=sources_k, batch_size=batch_size, shuffle=True)
                , prefix='train')
            , DataStreamMonitoring(
                  [cost]
                , DataETL.getFinalStream(teX, teY, sources=sources,
                      sources_k=sources_k, batch_size=batch_size, shuffle=True)
                , prefix='test')
            , UserFunc(save_function, after_epoch=True)
            , UserFunc(rank_em, after_epoch=True)
            , Printing()
            , LogToFile('logs/%s.csv' % name)
            ]
        )
    main_loop.run()

if __name__ == '__main__':
    trainencoder()