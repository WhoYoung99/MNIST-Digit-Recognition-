from __future__ import print_function

#### Libraries

# Standard library
import cPickle
import gzip
import os.path, os, struct, csv
import random
import math

# Third-party libraries
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def loadOneTrainingSet(filename, foldername):
    '''
    Return a 1-D List; which contains the target image's 28*28 bin array.
    '''
    os.chdir('/home/young/Desktop/%s' % foldername)
    f = open(filename, 'r')
    data = f.read()
    raw_image = list(struct.unpack('<784B', data))
    f.close()
    return raw_image

def loadTrainingFile(expanded_time, length=4000, filename='train.csv', foldername='train'):
    '''
    '''

    csv_content = tuple([line.rstrip().split(',') for line in open(filename)])
    shuffle_deck = np.arange(len(csv_content))
    shuffling = np.random.shuffle(shuffle_deck)
    valid_deck = shuffle_deck[:length]

    images, labels = [], []
    for index in shuffle_deck:
        fname, label = csv_content[index]
        image = loadOneTrainingSet(fname, foldername)
        image = np.array(image) / 255.0
        image = normalizeImage(image)
        images.append(image)
        labels.append(int(label))

        #for i in range(expanded_time):
        #    images.append(doDistortion(image))
        
        #labels += [int(label)] * expanded_time
    train_data = tuple([np.array(images, dtype='float32'), np.array(labels)])

    images, labels = [], []
    for index in valid_deck:
        fname, label = csv_content[index]
        image = loadOneTrainingSet(fname, foldername)
        image = np.array(image) / 255.0
        image = normalizeImage(image)
        images.append(image)
        labels.append(int(label))
    valid_data = tuple([np.array(images, dtype='float32'), np.array(labels)])
    os.chdir('/home/young/Desktop')
    return train_data, valid_data

def loadTestingFile(filename='sample.csv', foldername='test'):
    '''
    '''
    
    images = []
    with open(filename, 'r') as f:
        for line in f:
            fname = line.rstrip().split(',')[0]
            image = loadOneTrainingSet(fname, foldername)
            image = np.array(image) / 255.0
            #image[image < 0.1] = 0
            image = normalizeImage(image)
            images.append(image)
    f.close()
    os.chdir('/home/young/Desktop')
    return tuple([np.array(images, dtype='float32'), np.ones(50000,)*10])

def normalizeImage(src):
    '''
    For testing  & validating file normalize 
    '''
    to_black_threshold = 0.1
    skinny_threshold = 10

    image = np.copy(src).reshape(28, 28)
    image[image < to_black_threshold] = 0
    #bottom, top = np.min(np.nonzero(image)[0]), np.max(np.nonzero(image)[0])
    #left, right = np.min(np.nonzero(image.T)[0]), np.max(np.nonzero(image.T)[0])
    #bounding_box = image[bottom:top+1, left:right+1]
    #skinny = False
    #if left >= skinny_threshold: 
    #    skinny = True
    
    #if skinny:
    #    norm = np.copy(src)
    #else:
    norm = cv2.resize(image, (27, 27))
    norm = cv2.copyMakeBorder(norm, 1, 1, 1, 1, 0).reshape(-1,)
    return norm

def addElastic(norm, alpha=8, sigma=4, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
    """
    image = np.copy(norm)
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)

def addNoise(norm, noise_percent=0.10):
    '''
    norm.shape = (29, 29)
    return = (29, 29)
    '''
    noise = np.copy(norm)
    noise[np.random.randint(28, size=29*29*noise_percent), 
            np.random.randint(28, size=29*29*noise_percent)] = 0
    return noise

def addEnlarge(norm, enlarge_w=0.15, enlarge_h=0.15):
    '''
    norm.shape = (29, 29)
    return = (29, 29)
    '''
    image = np.copy(norm)
    skinny_threshold = 10
    bottom, top = np.min(np.nonzero(image)[0]), np.max(np.nonzero(image)[0])
    left, right = np.min(np.nonzero(image.T)[0]), np.max(np.nonzero(image.T)[0])
    bounding_box = image[bottom:top+1, left:right+1]
    h, w = bounding_box.shape
    en_w, en_h = int(round(w*(1+enlarge_w))), int(round(h*(1+enlarge_h)))
    if (29 - en_w) % 2 != 0:
        en_w += 1
    if (29 - en_h) % 2 != 0:
        en_h += 1

    b_w = (29 - en_w) / 2
    b_h = (29 - en_h) / 2
    #skinny = False
    #if left >= skinny_threshold: 
    #    skinny = True
    enlarge = cv2.resize(bounding_box, (en_w, en_h))
    enlarge = cv2.copyMakeBorder(enlarge, b_h, b_h, b_w, b_w, 0)
    #if skinny:
    #    enlarge = cv2.resize(image[bottom-1:top+2, 0:27], (29, 29))
    #else:
    #    enlarge = cv2.resize(bounding_box, (25, 25))
    #    enlarge = cv2.copyMakeBorder(enlarge, 2, 2, 2, 2, 0)
    return enlarge

def addRotation(norm, rotate_degree, scale=1):
    rangle = np.deg2rad(rotate_degree)
    image = np.copy(norm)
    

    w, h = image.shape
    # CALCULATE NEW IMAGE DIMENSIONS
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # GET ROTATION MATRIX
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), rotate_degree, scale)
    # OLD AND NEW CENTERS COMBINED WITH ROTATION
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
    # UPDATE TRANSLATION
    #rot_mat[0,2] += rot_move[0]
    #rot_mat[1,2] += rot_move[1]
    rotate = cv2.warpAffine(image, rot_mat, (int(math.ceil(nw)), 
                            int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

    #bottom, top = np.min(np.nonzero(image)[0]), np.max(np.nonzero(image)[0])
    #left, right = np.min(np.nonzero(image.T)[0]), np.max(np.nonzero(image.T)[0])
    #bounding_box = image[bottom:top+1, left:right+1]

    '''
    en_h, en_w = rotate.shape
    if (29 - en_w) % 2 != 0:
        en_w += 1
        pick = [(29-en_w)/2, (29-en_w)/2 +1]
        random.shuffle(pick)
        b_wLeft, b_wRight = pick
    else:
        b_wLeft, b_wRight = (29-en_w)/2, (29-en_w)/2
    if (29 - en_h) % 2 != 0:
        en_h += 1
        pick = [(29-en_h)/2, (29-en_h)/2 +1]
        random.shuffle(pick)
        b_hTop, b_hBot = pick
    else:
        b_hTop, b_hBot = (29-en_h)/2, (29-en_h)/2
    '''
    rotate = cv2.resize(rotate, (29, 29))
    #rotate = cv2.copyMakeBorder(rotate, b_hTop, b_hBot, b_wLeft, b_wRight, 0)
    return rotate

def doDistortion(src):
    norm = src.reshape(29,29)
    temp = addElastic(norm, alpha=random.choice([8,10,12]), 
                        sigma=random.choice([4,6,8]))
    temp = addNoise(temp, noise_percent=random.uniform(0, .07))
    temp = addEnlarge(temp, enlarge_w=random.uniform(-.15, .15), 
                        enlarge_h=random.uniform(-.15, .15))
    imageDistor = addRotation(temp, rotate_degree=random.randint(-15, 15))
    return imageDistor.reshape(-1,)

def distortionImage(src):
    '''
    Handle all preprocess: normalize, inject noise, enlarge, rotate
    input: 1 image
    output: 8 images, include "norm", "noise", "enlarge", "rotate"*2, "elastic"*3
    '''
    to_black_threshold = 0.1
    skinny_threshold = 10
    noise_percent = 0.15
    rotate_degree, scale = [10, -10], 1

    ### Normalization & Enlarge ####################################################
    image = np.copy(src).reshape(28, 28)
    image[image < to_black_threshold] = 0
    bottom, top = np.min(np.nonzero(image)[0]), np.max(np.nonzero(image)[0])
    left, right = np.min(np.nonzero(image.T)[0]), np.max(np.nonzero(image.T)[0])
    bounding_box = image[bottom:top+1, left:right+1]

    skinny = False
    if left >= skinny_threshold: 
        skinny = True
    if skinny:
    #    norm = np.copy(src)
        enlarge = cv2.resize(image[bottom-1:top+2, 0:27], (29, 29)).reshape(-1,)
    else:
    #    norm = cv2.resize(bounding_box, (26, 26))
    #    norm = cv2.copyMakeBorder(norm, 1, 1, 1, 1, 0).reshape(-1,)
        enlarge = cv2.resize(bounding_box, (25, 25)).reshape(-1,)
        enlarge = cv2.copyMakeBorder(enlarge, 2, 2, 2, 2, 0).reshape(-1,)
    norm = normalizeImage(image)
    
    ### Inject Noise ##############################################################
    noise = np.copy(norm).reshape(29, 29)
    noise[np.random.randint(28, size=29*29*noise_percent), 
            np.random.randint(28, size=29*29*noise_percent)] = 0
    noise = noise.reshape(-1,)

    ### Rotate (and Translate) ####################################################
    two_side = []
    for angle in rotate_degree:
        rangle = np.deg2rad(angle) # angle in rads
        w, h = 29, 29
        # CALCULATE NEW IMAGE DIMENSIONS
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # GET ROTATION MATRIX
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # OLD AND NEW CENTERS COMBINED WITH ROTATION
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
        # UPDATE TRANSLATION
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        rotate = cv2.warpAffine(np.copy(norm).reshape(29, 29), rot_mat, (int(math.ceil(nw)), 
                                    int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        rotate = cv2.resize(rotate, (29, 29)).reshape(-1,)
        two_side.append(rotate)

    rotate_R, rotate_L = two_side
    ### Elastic transform ##########################################################
    image = np.copy(norm).reshape(29, 29)
    elastic1 = elastic_transform(image, alpha=36, sigma=6, random_state=None).reshape(-1,)
    elastic2 = elastic_transform(image, alpha=50, sigma=5, random_state=None).reshape(-1,)
    elastic3 = elastic_transform(image, alpha=60, sigma=4, random_state=None).reshape(-1,)

    return norm, noise, enlarge, rotate_R, rotate_L, elastic1, elastic2, elastic3

def doPreprocess(expanded_time=15, length=4000):
    print("Preparing Testing Set...")
    ### Preparing Testing Set: 50,000
    if os.path.exists("../data/mnist_testing.pkl.gz"):
        pass
    else:
        test = loadTestingFile()
        f = gzip.open("../data/mnist_testing.pkl.gz", "w")
        cPickle.dump((test), f)
        f.close()
    print("Preparing Training and Validating Set")
    ### Preparing Training and Validating Set: 
    if os.path.exists("../data/mnist_training%s.pkl.gz" %(10000)):
        #os.remove("../data/mnist_training%s.pkl.gz" %(10000))
        pass
    else:
        train, valid = loadTrainingFile(expanded_time=50)
        f = gzip.open("../data/mnist_training%s.pkl.gz" %(10000), "w")
        cPickle.dump((train), f)
        f.close()
        f = gzip.open("../data/mnist_validating%s.pkl.gz" %(length), "w")
        cPickle.dump((valid), f)
        f.close()

    #'''
    ### Preparing Expanded Training Set: length*5*5
    print("Preparing Expanded Training Set")
    if os.path.exists("../data/mnist_expanded%s.pkl.gz" %(10000*expanded_time)):
        #os.remove("../data/mnist_expanded.pkl.gz")
        pass
    else:
        f = gzip.open("../data/mnist_training%s.pkl.gz" %(10000), 'rb')
        training_data = cPickle.load(f)
        f.close()
        expanded_training_pairs = []
        j = 0 # counter
        for x, y in zip(training_data[0], training_data[1]):
            expanded_training_pairs.append((x, y))
            image = np.reshape(x, (-1, 29))
            j += 1
            # if j % 1000 == 0: print("Expanding image number", j)
            # iterate over data telling us the details of how to
            # do the displacement
            '''
            for d, axis, index_position, index in [
                    (1,  0, "first", 0),
                    (-1, 0, "first", 27),
                    (1,  1, "last",  0),
                    (-1, 1, "last",  27)]:
                new_img = np.roll(image, d, axis)
                if index_position == "first": 
                    new_img[index, :] = np.zeros(28)
                else: 
                    new_img[:, index] = np.zeros(28)
            '''
            for i in xrange(expanded_time):    
                new_img = doDistortion(image)
                expanded_training_pairs.append((np.reshape(new_img, 841), y))
        random.shuffle(expanded_training_pairs)
        expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
        print("Saving expanded data. This may take a few minutes.")
        f = gzip.open("../data/mnist_expanded%s.pkl.gz" %(10000*expanded_time), "w")
        cPickle.dump((expanded_training_data), f)
        f.close()
    
    print("Finishing Processing... ")    


if __name__ == '__main__':
    train_length = 2000

    if os.path.exists("../data/mnist_testing.pkl.gz"):
        print("The testing set already exists.")
    else:
        test = loadTestingFile()
        f = gzip.open("../data/mnist_testing.pkl.gz", "w")
        cPickle.dump((test), f)
        f.close()

    if os.path.exists("../data/mnist_training%s_NER.pkl.gz" %train_length):
        print("The training set already exists.")
    else:
        train, valid = loadTrainingFile(train_length)
        f = gzip.open("../data/mnist_training%s_NER.pkl.gz" %(train_length*5), "w")
        cPickle.dump((train), f)
        f.close()
        f = gzip.open("../data/mnist_validating%s.pkl.gz" %(10000-train_length), "w")
        cPickle.dump((valid), f)
        f.close()

    if os.path.exists("../data/mnist_expanded%s_NER.pkl.gz" %(train_length*25)):
        print("The expanded training set already exists.  Exiting.")
    else:
        f = gzip.open("../data/mnist_training%s_NER.pkl.gz" %(train_length*5), 'rb')
        training_data = cPickle.load(f)
        f.close()
        expanded_training_pairs = []
        j = 0 # counter
        for x, y in zip(training_data[0], training_data[1]):
            expanded_training_pairs.append((x, y))
            image = np.reshape(x, (-1, 28))
            j += 1
            if j % 1000 == 0: print("Expanding image number", j)
            # iterate over data telling us the details of how to
            # do the displacement
            for d, axis, index_position, index in [
                    (1,  0, "first", 0),
                    (-1, 0, "first", 27),
                    (1,  1, "last",  0),
                    (-1, 1, "last",  27)]:
                new_img = np.roll(image, d, axis)
                if index_position == "first": 
                    new_img[index, :] = np.zeros(28)
                else: 
                    new_img[:, index] = np.zeros(28)
                expanded_training_pairs.append((np.reshape(new_img, 784), y))
        random.shuffle(expanded_training_pairs)
        expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
        print("Saving expanded data. This may take a few minutes.")
        f = gzip.open("../data/mnist_expanded%s_NER.pkl.gz" %(train_length*25), "w")
        cPickle.dump((expanded_training_data), f)
        f.close()