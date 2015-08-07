import os
import csv
import operator

import numpy as np
import pandas as pd
import theano
from sklearn.cross_validation import train_test_split

from fuel import config
from fuel.transformers import Transformer
from foxhound.utils import shuffle
from pycocotools.coco import COCO
from config import COCO_DIR, SBU_DIR, FLICKR8K_DIR, FLICKR30K_DIR

def coco(mode="dev", n_captions=1, test_size=None):
    """loads coco data into train and test features and targets.
    mode = 'dev' is used for development (quick load of subset)
    """
    # train_fns
    dataType='train2014'
    train_fns = os.listdir("%s/features/%s"%(COCO_DIR, dataType))

    # reduce it to a dev set
    if mode == "dev":
        train_fns = shuffle(train_fns)[:256]
    trX, trY = loadFeaturesTargets(train_fns, dataType, n_captions)

    # val_fns
    dataType='val2014'
    test_fns = os.listdir("%s/features/%s"%(COCO_DIR, dataType))

    # reduce it to a dev set
    if mode == "dev":
        test_fns = shuffle(test_fns)[:128]

    if test_size:
        test_fns = shuffle(test_fns)[:test_size]

    teX, teY = loadFeaturesTargets(test_fns, dataType, n_captions)

    return trX, teX, trY, teY

def sbu(mode="dev", n_sbu=None, test_size=None):
    """loads sbu data into train and test features and targets.
    mode = 'dev' is used for development (quick load of subset)
    """
    if mode == 'dev':
        n_sbu = 256
        test_size = 128

    sbuX, sbuY, _ = sbuXYFilenames(n_sbu)
    return train_test_split(sbuX, sbuY, test_size=test_size)

def flickr8kXYFilenames(n_examples=None):
    feature_path = os.path.join(FLICKR8K_DIR, "features")
    caption_path = os.path.join(FLICKR8K_DIR, "annotations/Flickr8k.token.txt")

    f = open(caption_path, 'r')
    lines = f.read().splitlines()

    data_dict = {}

    def parse(line):
        split = line.split('#')
        img = split[0]
        caption_ugly = split[1:]
        if isinstance(caption_ugly, list):
            caption_ugly = ' '.join(caption_ugly)
        i, caption = caption_ugly.split('\t')
        handful = data_dict.get(img, [])
        handful.append(caption)
        data_dict[img] = handful

    print("Parsing flickr8k captions...")
    for i, line in enumerate(lines):
        parse(line)

    fns, Y = zip(*data_dict.items())
    X = []
    successes = []

    print("Loading flickr8k features...")
    for i in range(len(fns)):
        fn = fns[i]
        try:
            name = fn.split('.')[0]
            X.append(np.load(os.path.join(feature_path, "%s.npy" % name)))
            successes.append(i)
        except Exception, e:
            continue

    # get only the successful funs
    if successes:
        Y = operator.itemgetter(*successes)(Y)
        fns = operator.itemgetter(*successes)(fns)

    return X, Y, fns

def sbuXYFilenames(n_examples=None):
    """
    n_examples to try to load.  It might not load all of them
    """
    sbu_feature_path = os.path.join(SBU_DIR, "features")
    sbu_caption_path = os.path.join(SBU_DIR, "SBU_captioned_photo_dataset_captions.txt")
    fns = os.listdir(sbu_feature_path)
    if n_examples:
        fns = fns[:n_examples]

    print "Reading SBU captions"
    f = open(sbu_caption_path, 'rb')
    captions = f.read().splitlines()
    f.close()

    X, Y = [], []
    successes = []

    print "Loading in SBU Features"
    for i in range(len(fns)):
        fn = fns[i]
        try:
            # fn should be SBU_%d
            index = int(fn[4:].split(".")[0])
            X.append(np.load(os.path.join(sbu_feature_path, fn)))
            Y.append([captions[index]])
            successes.append(i)
        except:
            continue

    # get only the successful fns
    fns = operator.itemgetter(*successes)(fns)
    print "SBU Done!"

    return X, Y, fns

def cocoXYFilenames(n_captions=5, dataType='val2014'):
    """Helps when you are evaluating and want the filenames
    associated with the features and target variables

    Parameters
    ----------
    n_captions: integer
        how many captions to load for the image

    dataType: 'val2014' or 'train2014'

    Returns
    -------
    X: the features
    Y: the targets
    filenames: the filenames corresponding to each
    """
    fns = os.listdir("%s/features/%s"%(COCO_DIR, dataType))
    fns = shuffle(fns)
    X, Y = loadFeaturesTargets(fns, dataType, n_captions)

    return X, Y, fns

def loadFeaturesTargets(fns, dataType, n_captions=1):
    """
    Note: filenames should come from the same type of dataType.

    filenames from val2014, for example, should have dataType val2014
    Parameters
    ----------
    fns: filenames, strings

    dataType: string folder, i.e. train2014, val2014

    n_captions: int, number of captions for each image to load

    Returns
    -------
    X: list of im_vects
        1st list length = len(fns)
        vectors are shape (4096, )

    Y: list of list of captions.
        1st list length = len(fns)
        sublist length = n_captions
    """
    annFile = '%s/annotations/captions_%s.json'%(COCO_DIR,dataType)
    caps=COCO(annFile)

    X = []
    Y = []

    for fn in fns:
        # Features
        x = np.load('%s/features/%s/%s'%(COCO_DIR, dataType, fn))

        # Targets
        annIds = caps.getAnnIds(imgIds=getImageId(fn));
        anns = caps.loadAnns(annIds)

        # sample n_captions per image
        anns = shuffle(anns)
        captions = [getCaption(anns[i]) for i in range(n_captions)]

        X.append(x)
        Y.append(captions)

    return X, Y

def getImageId(fn):
    """Filename to image id

    Parameters
    ----------
    fn: a string
        filename of the COCO dataset.

        example:
        COCO_val2014_000000581929.npy

    Returns
    imageId: an int
    """
    return int(fn.split("_")[-1].split('.')[0])

def getCaption(ann):
    """gets Caption from the COCO annotation object

    Parameters
    ----------
    ann: list of annotation objects
    """
    return str(ann["caption"])

def fillOutFilenames(filenames, n_captions):
    new_fns = []
    for fn in filenames:
        new_fns.extend([fn for i in range(n_captions)])
    return new_fns

# Foxhound + Fuel
class FoxyDataStream(object):
    """FoxyDataStream attempts to merge the gap between fuel DataStreams and
    Foxhound iterators.

    The place we will be doing this merge is in the blocks MainLoop. Inserting
    a FoxyDataStream() in place of a DataStream.default_stream()
    will suffice.

    (Note)
    These are broken down into the following common areas
    - dataset which has (features, targets) or (X, Y)
    - iteration_scheme (sequential vs shuffling, batch_size)
    - transforms

    Parameters
    ----------
    data: tuple of X, Y

    sources: tuple of sourcenameX, sourcenameY

    iterator: a Foxhound iterator.  The use is jank right now, but always use
        trXt and trYt as the X and Y transforms respectively
    """

    def __init__(self, data, sources, make_iterator, iteration_scheme=None):
        self.data = data
        self.sources = sources
        self.iterator = make_iterator
        # self.iterator_prototype = make_iterator
        self.iteration_scheme = iteration_scheme # Compatibility with the blocks mainloop

    def get_epoch_iterator(self, as_dict=False):

        # iterator = self.iterator_prototype(None)
        # print iterator
        for datamb in self.iterator.iterXY(*self.data):
            yield dict(zip(self.sources, datamb)) if as_dict else datamb

class FoxyIterationScheme(object):
    """mimics like a Fox a fuel iteration scheme

    Important Attributes
    --------------------
    num_batches: int

    OR

    num_examples: int

    batch_size: int
    """
    def __init__(self, examples, batch_size):
        self.num_examples = examples
        self.batch_size = batch_size

class GloveTransformer(Transformer):
    glove_folder = "glove"
    vector_dim = 0

    def __init__(self, glove_file, data_stream, vectorizer):
        super(GloveTransformer, self).__init__(data_stream)
        dir_path = os.path.join(config.data_path, self.glove_folder)
        data_path = os.path.join(dir_path, glove_file)
        raw = pd.read_csv(data_path, header=None, sep=' ', quoting=csv.QUOTE_NONE, nrows=50000)
        #raw = pd.read_csv(data_path, nrows=400, header=None, sep=' ', quoting=csv.QUOTE_NONE)
        keys = raw[0].values
        self.vectors = raw[range(1, len(raw.columns))].values.astype(theano.config.floatX)
        self.vector_dim = self.vectors.shape[1]

        # lookup will have (key, val) -> (word-string, row index in self.vectors)
        row_indexes = range(self.vectors.shape[0])
        self.lookup = dict(zip(keys, row_indexes))
        self.reverse_lookup = dict(zip(row_indexes, keys))
        self.vectorizer = vectorizer

    def get_data(self, request=None):
        if request is not None:
            raise ValueError

        image_reps, codes = next(self.child_epoch_iterator)

        def process_tokens(tokens):
            output = np.random.rand(len(tokens), self.vector_dim)
            for i,t in enumerate(tokens):
                word = self.vectorizer.decoder[t]
                if word in self.lookup:
                    output[i, :] = self.vectors[self.lookup[word]]
                # else t is UNK so we leave the output alone
            return output

        word_reps = np.asarray(
              [process_tokens(tokens) for tokens in codes.T]
            , dtype=theano.config.floatX)

        return image_reps, word_reps

class ShuffleBatch(Transformer):
    """Shuffle the Batch, helpful when generating contrastive examples"""
    def __init__(self, data_stream):
        super(ShuffleBatch, self).__init__(data_stream)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)
        return shuffle(*data)
