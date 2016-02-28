##Requirements
------------
- Numpy
- Pandas
- Matplotlib
- Theano
- Blocks
- Fuel
- sklearn
- IPython
- Foxhound
- pycocotools

##Pre-trained Models
--------------------
###GloVe Vector Files
In your fuel config file (i.e. ```~/.fuelrc```), ```data_path``` should be set to a datasets directory (i.e. ```~/datasets```). In this directory, you should create a folder to store the GloVe vector file named ```glove```.  I am using the GloVe Vectors with 6 billion tokens and 300 dimensional vectors, trained on the Gigaword + Wikipedia corpus, [download link can be found on Glove website](http://www-nlp.stanford.edu/projects/glove).  If you did it all correct, the path to the GloVe vectors should look like ```~/datasets/glove/glove.6B.300d.txt.gz```.

###Image Features
You can precompute image features for all the image data using the IndicoAPI (wait for easynet support, which gives back the 4096 length vectors which you want!)

##Training Data
---------------
###MSCOCO - Microsoft Common-Objects in Context
You define where your MSCOCO data is stored through the ```config.py``` file.  In my config.py, ```COCO_DIR=~/datasets/coco```.

You should have precomputed image features for MSCOCO stored in this directory, at the location ```$COCO_DIR/features/train2014```, or ```$COCO_DIR/features/val2014```.  The images should be named exactly as the image files, with ```.jpg``` extension subsituted with ```.npy```.

Caption files should be located in ```$COCO_DIR/annotations```. For example, captions for the train2014 data should be located at ```$COCO_DIR/annotations/captions_train2014.json```.

###FOR SBU (SHOULD I EVEN INCLUDE SBU?! IT"S SUCH A BAD DATASET...)
Must have image features in the ```SBU_DIR/features``` ordered in line with the captions.  Caption file should be located at ```SBU_DIR/SBU_captioned_photo_dataset_captions.txt```
