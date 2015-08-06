# common python
import decimal
from collections import OrderedDict

# scientific python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# foxhound
from foxhound.preprocessing import Tokenizer
from foxhound import iterators
from foxhound.theano_utils import floatX, intX
from foxhound.transforms import SeqPadded
from foxhound.rng import py_rng

# fuel
from fuel.transformers import Merge

# local imports
from dataset import (coco, cocoXYFilenames, FoxyDataStream, GloveTransformer,
    ShuffleBatch, FoxyIterationScheme, loadFeaturesTargets, fillOutFilenames,
    sbuXYFilenames)
from utils import dict2json, vStackMatrices, DecimalEncoder, ModelIO

def sampleCaptions(ymb, K=1):
    """ymb = minibatch of captions
    it samples K captions from the available list of n captions
    """
    sampled_captions = []
    for captions_of_an_img in ymb:
        sampled_captions.extend(py_rng.sample(captions_of_an_img, K))
    return sampled_captions

def concatCaptions(ymb, K=5):
    """ymb = minibatch of captions
    it concatenates the first K captions from the available list of n captions.
    While destorying some sentence order when concatenating, this is
    helpful when we want the presence of a token"""

    def joinListOfCaptions(listOfCaptions):
        return " ".join(listOfCaptions[:K])
    return map(joinListOfCaptions, ymb)

def prepVect(min_df=2, max_features=50000, n_captions=5, n_sbu=None,
             multilabel=False):
    print "prepping the Word Tokenizer..."
    _0, _1, trY, _3 = coco(mode='full', n_captions=n_captions)
    if n_sbu:
        _4, sbuY, _5 = sbuXYFilenames(n_sbu)
        trY.extend(sbuY)
    vect = Tokenizer(min_df=min_df, max_features=max_features)
    captions = sampleCaptions(trY, n_captions)
    vect.fit(captions)
    if multilabel:
        mlb = MultiLabelBinarizer()
        mlb.fit(vect.transform(captions))
        return vect, mlb
    # if not multilabel:
    return vect

dataset_name = 'coco_train2014+sbu100000'

# global vectorizer
vect_name = 'tokenizer_%s' % dataset_name
mlb_name = 'mlb_%s' % dataset_name
try:
    if mlb_name:
        mlb = ModelIO.load(mlb_name)
        print "MLB loaded from file"
    vect = ModelIO.load(vect_name)
    # vect = ModelIO.load('tokenizer_reddit') # gloveglove
    print "Tokenizer loaded from file."
except:
    if mlb_name:
        vect, mlb = prepVect(n_sbu=100000, n_captions=1, multilabel=True)
        ModelIO.save(vect, vect_name)
        ModelIO.save(mlb, mlb_name)
        print "Saved %s, %s for future use." % (vect_name, mlb_name)
    else:
        vect = prepVect(n_sbu=100000, n_captions=1)
        ModelIO.save(vect, vect_name)
        print "Saved %s for future use." % vect_name

class DataETL():

    @staticmethod
    def getFinalStream(X, Y, sources, sources_k, batch_size=128, embedding_dim=300,
        shuffle=False):
        """
        Returns
        -------
        merged stream with sources = sources + sources_k
        """
        trX, trY = (X, Y)
        trX_k, trY_k = (X, Y)

        # Transforms
        trXt=lambda x: floatX(x)
        Yt=lambda y: intX(SeqPadded(vect.transform(sampleCaptions(y)), 'back'))

        # Foxhound Iterators
        # RCL: Write own iterator to sample positive examples/captions, since there are 5 for each image.
        train_iterator = iterators.Linear(
            trXt=trXt, trYt=Yt, size=batch_size, shuffle=shuffle
            )
        train_iterator_k = iterators.Linear(
            trXt=trXt, trYt=Yt, size=batch_size, shuffle=True
            )

        # FoxyDataStreams
        train_stream = FoxyDataStream(
              (trX, trY)
            , sources
            , train_iterator
            , FoxyIterationScheme(len(trX), batch_size)
            )

        train_stream_k = FoxyDataStream(
              (trX_k, trY_k)
            , sources_k
            , train_iterator_k
            , FoxyIterationScheme(len(trX), batch_size)
            )
        glove_version = "glove.6B.%sd.txt.gz" % embedding_dim
        train_transformer = GloveTransformer(
            glove_version, data_stream=train_stream, vectorizer=vect
            )
        train_transformer_k = GloveTransformer(
            glove_version, data_stream=train_stream_k, vectorizer=vect
            )

        # Final Data Streams w/ contrastive examples
        final_train_stream = Merge(
              (train_transformer, ShuffleBatch(train_transformer_k))
            , sources + sources_k
            )
        final_train_stream.iteration_scheme = FoxyIterationScheme(len(trX), batch_size)

        return final_train_stream


class ModelEval():

    @staticmethod
    def rankcaptions(filenames, top_n=5):
        # n_captions = top_n # the captions it ranks as highest should all be relevant
        n_captions = 1 # RCL: image caption mismatch when n_captions is not just one
        batch_size = 128
        image_features, captions = loadFeaturesTargets(filenames, 'val2014', n_captions=n_captions)
        stream = DataETL.getFinalStream(
              image_features
            , captions
            , ("image_vects", "word_vects")
            , ("image_vects_k", "word_vects_k")
            , batch_size=batch_size
            )

        f_emb = ModelIO.load('/home/luke/datasets/coco/predict/fullencoder_maxfeatures.50000')
        im_emb, s_emb = None, None
        print "Computing Image and Text Embeddings"
        for batch in stream.get_epoch_iterator():
            im_vects = batch[0]
            s_vects = batch[1]
            batch_im_emb, batch_s_emb = f_emb(im_vects, s_vects)
            im_emb = vStackMatrices(im_emb, batch_im_emb)
            s_emb = vStackMatrices(s_emb, batch_s_emb)

        # account for make sure theres matching fns for each of the n_captions
        image_fns = fillOutFilenames(filenames, n_captions=n_captions)

        print "Computing Cosine Distances and Ranking Captions"
        relevant_captions = ModelEval.getRelevantCaptions(
            im_emb, s_emb, image_fns, captions, z=n_captions, top_n=top_n
        )
        dict2json(relevant_captions, "rankcaptions_fullencoder_maxfeatures.50000.json", cls=DecimalEncoder)
        return relevant_captions

    @staticmethod
    def rankscores(final_train_stream, final_test_stream, f_emb):

        i2t = ModelEval.i2t
        train_ep = final_train_stream.get_epoch_iterator()
        test_ep = final_test_stream.get_epoch_iterator()

        train_metrics = []
        test_metrics = []
        for train_data, test_data in train_ep, test_ep:
            im_emb, s_emb = f_emb(*train_data)
            train_metrics.append(i2t(im_emb, s_emb))
            im_emb, s_emb = f_emb(*train_data)
            test_metrics.append(i2t(im_emb, s_emb))
        train_metrics = np.vstack(train_metrics)
        test_metrics = np.vstack(test_metrics)

        metric_names = ("r1", "r5", "r10", "med")
        print "\nMean Metric Scores:"
        for i, metric_name in enumerate(metric_names):
            for metrics in (train_metrics, test_metrics):
                print "%s: %d" % metric_name, np.mean(metrics[:, i])

        return train_metrics, test_metrics

    @staticmethod
    def i2t(images, captions, z=1, npts=None):
        """
        Taken from https://github.com/ryankiros/skip-thoughts/blob/master/eval_rank.py
        Images: (z*N, K) matrix of image embeddings
        Captions: (z*N, K) matrix of caption embeddings
        """
        if npts == None:
            npts = images.shape[0] / z
        index_list = []

        # Project captions
        for i in range(len(captions)):
            captions[i] /= np.linalg.norm(captions[i])

        ranks = np.zeros(npts)
        for index in range(npts):

            # Get query image
            im = images[z * index].reshape(1, images.shape[1])
            im /= np.linalg.norm(im)

            # Compute scores
            d = np.dot(im, captions.T).flatten()
            inds = np.argsort(d)[::-1]
            index_list.append(inds[0])

            # Score
            rank = 1e20
            for i in range(z*index, z*index + z, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        return (r1, r5, r10, medr)

    @staticmethod
    def t2i(images, captions, z=1, npts=None):
        """
        Taken from https://github.com/ryankiros/skip-thoughts/blob/master/eval_rank.py
        Images: (z*N, K) matrix of image embeddings
        Captions: (z*N, K) matrix of captions embeddings
        """
        if npts == None:
            npts = images.shape[0] / z
        ims = np.array([images[i] for i in range(0, len(images), z)])


        # Project images
        for i in range(len(ims)):
            ims[i] /= np.linalg.norm(ims[i])

        # Project captions
        for i in range(len(captions)):
            captions[i] /= np.linalg.norm(captions[i])

        ranks = np.zeros(z * npts)
        for index in range(npts):

            # Get query captions
            queries = captions[z*index : z*index + z]

            # Compute scores
            d = np.dot(queries, ims.T)
            inds = np.zeros(d.shape)
            for i in range(len(inds)):
                inds[i] = np.argsort(d[i])[::-1]
                ranks[z * index + i] = np.where(inds[i] == index)[0][0]

        # Compute metrics
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        return (r1, r5, r10, medr)

    @staticmethod
    def ImageSentenceRanking(images, captions, z=1):
        """
        Print nicely formatted tables each iteration
        N = 1000 is commonly used.
        images: (N, K)
        captions: (N, K)
        z: number of images per caption (see i2t, t2i)
        """
        rank_labels = ('R @ 1', 'R @ 5', 'R @ 10', 'Med R')
        image_annotation = ModelEval.i2t(images, captions, z)
        image_search = ModelEval.t2i(images, captions, z)

        print pd.DataFrame(OrderedDict(zip(rank_labels, image_annotation)), 
            index=pd.Index(["Image Annotation"])).to_string()
        print pd.DataFrame(OrderedDict(zip(rank_labels, image_search)), 
            index=pd.Index(["Image Search    "])).to_string()

        return image_annotation, image_search

    @staticmethod
    def ImageSearchSingleCategory(ims, mlb_matrix, captions, category_key, thresh):
        """do a single category, like dog"""
        # project images
        for i in range(len(ims)):
            ims[i] /= np.linalg.norm(ims[i])
        
        # project single captions
        for i in range(len(captions)):
            captions[i] /= np.linalg.norm(captions[i])

        assert captions.shape[0] == 1

        sims = np.dot(captions, ims.T).flatten()
        
        found = []
        n_found = 0
        n_matches = 0
        for i in range(len(sims)):
            if sims[i] > thresh:
                n_found += 1
                found.append((i, sims[i]))

                # depends on category_key being a single integer
                if mlb_matrix[i][category_key]:
                    n_matches += 1

        print "n_found: ", n_found
        print "n_matches: ", n_matches
        return found

    @staticmethod
    def getRelevantCaptions(im_emb, s_emb, image_fns, caption_strings, top_n, z=1, npts=None):
        """
        parameters
        ----------
        Images: (z*N, K) matrix of im_emb
        Captions: (z*N, K) matrix of captions
        image_fns: the filenames of im_emb for each image vectors in the im_emb matrix
        captions_strings: the captions (as strings) for each sentence vector in captions matrix

        Returns
        -------
        relevant_captions: dictionary storing the top_n rank predictions for each image file

        looks like
        {
            ... , 
            filepath.npy: {
                captions: ["caption with ranking 1", ...]
                cos_sims: [0.9, 0.5, ...]
            },
            ...
        }
        """
        if npts == None:
            npts = im_emb.shape[0] / z

        relevant_captions = {}

        # Project captions
        for i in range(len(s_emb)):
            s_emb[i] /= np.linalg.norm(s_emb[i])

        for index in range(npts):

            # Get query image
            im = im_emb[z * index].reshape(1, im_emb.shape[1])
            im /= np.linalg.norm(im)

            # Compute scores
            d = np.dot(im, s_emb.T).flatten() # cosine distance
            inds = np.argsort(d)[::-1] # sort by highest cosine distance

            # build up relevant top_n captions
            image_fn = image_fns[index]
            top_inds = inds[:top_n]
            top_captions = [caption_strings[ind] for ind in top_inds]
            top_cos_sims = [decimal.Decimal(float(d[ind])) for ind in top_inds]

            relevant_captions[image_fn] = {
                  "captions": top_captions
                , "cos_sims": top_cos_sims
                }

        return relevant_captions

    @staticmethod
    def rank_function(self=None):
        teX, teY, _ = cocoXYFilenames(n_captions=5)
        sources = ('X', 'Y')
        sources_k = ('X_k', 'Y_k')
        stream = DataETL.getFinalStream(teX, teY, sources=sources,
                            sources_k=sources_k, batch_size=1000,
                            shuffle=False)
        images, captions, _0, _1 = stream.get_epoch_iterator().next()

        predict_dir = '/home/luke/datasets/coco/predict/'
        # encoder_name = '+coco_encoder_lstm_dim.300'
        encoder_name = 'sbu.100000+coco_encoder_lstm_dim.300_adadelta'
        # encoder_name = 'fullencoder_maxfeatures.50000_epochsampler'
        f_emb = ModelIO.load(predict_dir + encoder_name)
        image_embs, caption_embs = f_emb(images, captions)
        ModelEval.ImageSentenceRanking(image_embs, caption_embs)