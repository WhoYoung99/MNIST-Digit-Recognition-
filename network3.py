"""network3.py
~~~~~~~~~~~~~~
A Theano-based program for training and running simple neural
networks.
Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).
When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.
Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.
This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).
"""

#### Libraries
# Standard library
import cPickle
import gzip
import os, struct
from time import strftime

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from preprocess import *


#### Constants
GPU = True
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True."

#### Load the MNIST data
def load_data_shared(expanded_time, length=4000):
    prework = doPreprocess(expanded_time=expanded_time, length=length)
    
    ###filename="../data/mnist_training%s.pkl.gz" %length

    ###filename="../data/mnist_expanded%s.pkl.gz" %(length*5)
    filename="../data/mnist_expanded%s.pkl.gz" %(10000*expanded_time)
    f = gzip.open(filename, 'rb')
    expand_training_data = cPickle.load(f)
    f.close()
    #print(len(expand_training_data[0][0]))

    filename="../data/mnist_training%s.pkl.gz" %(10000)
    f = gzip.open(filename, 'rb')
    training_data = cPickle.load(f)
    f.close()
    #print(len(training_data[0][0]))

    filename="../data/mnist_validating%s.pkl.gz" %(length)
    f = gzip.open(filename, 'rb')
    validation_data = cPickle.load(f)
    f.close()
    #print(len(validation_data[0][0]))

    filename="../data/mnist_testing.pkl.gz"
    f = gzip.open(filename, 'rb')
    test_data = cPickle.load(f)
    f.close()
    #print(len(test_data[0][0]))

    #training_data = loadTrainingFile(0, 2000)
    #validation_data = loadTrainingFile(2000, 6000)
    #test_data = loadTrainingFile(6000,10000)

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(expand_training_data), shared(training_data), 
                 shared(validation_data), shared(test_data)]


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

def loadTrainingFile(p1, p2, filename='train.csv', foldername='train'):
    '''
    Reading all training samples' file name, and their corrosponding labels.
    Return: a list with "Image" objects
    '''
    print 'Loading training set...'
    images = []
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            fname, label = line.rstrip().split(',')
            image = loadOneTrainingSet(fname, foldername)
            image = np.array(image) / 255.0
            images.append(image)
            labels.append(int(label))
    f.close()
    os.chdir('/home/young/Desktop')
    return tuple([np.array(images, dtype='float32')[p1:p2], np.array(labels)[p1:p2]])

def loadTestingFile(filename='sample.csv', foldername='test'):
    '''
    Return: a list with "Image" objects
    '''
    print 'Loading testing set...'
    images = []
    with open(filename, 'r') as f:
        for line in f:
            fname = line.rstrip().split(',')[0]
            image = loadOneTrainingSet(fname, foldername)
            image = np.array(image) / 255.0
            images.append(image)
    f.close()
    os.chdir('/home/young/Desktop')
    return tuple([np.array(images, dtype='float32'), np.ones(50000,)*10])

#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        ############################################
        self.test_mb_predictions_prob = theano.function(
            #[i], self.layers[-1].y_out,
            ################################
            [i], self.layers[-1].y_out_prob,
            ################################
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        #############################################

        # Do the actual training
        #early_stop = 0
        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            #if early_stop > 20:
            #    print("Early stop at epoch #{0}".format(epoch-1))
            #    break
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                #if iteration % 1000 == 0:
                #    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        #early_stop = 0
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        #if test_data:
                        #    test_accuracy = np.mean(
                        #        [test_mb_accuracy(j) for j in xrange(num_test_batches)])
                        #    print('The corresponding test accuracy is {0:.2%}'.format(
                        #        test_accuracy))
                    #else:
                    #    early_stop += 1

        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        #print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))
        ############
        prob_result = np.array([self.test_mb_predictions_prob(j) for j in xrange(num_test_batches)]).reshape(-1, 10)
        digit_result = np.array([self.test_mb_predictions(j) for j in xrange(num_test_batches)]).reshape(-1,)
        #print(prob_result[:10])
        #return prob_result

        return digit_result, prob_result
        ############


#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.
    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.
        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.
        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        #############################
        self.y_out_prob = self.output
        #############################
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))
    
    def getResult(self):
        return self.y_out

#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)

def chooseVoting(voting_list):
    counting = Counter(voting_list)
    total_weight = sum(counting.values())
    empty = np.zeros(10)
    for key in counting.keys():
        empty[key] = str(round(counting[key]/float(total_weight), 4))
    return list(empty)

def choosePVoting(voting_p_list):
    '''
    input shape: (voters, 10)
    '''
    l = float(len(voting_p_list))
    result = np.sum(voting_p_list, axis=0)/l
    result[result < 0.1] = 0
    #result = ("%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f" %tuple([i for i in result])).split(',')
    result = "{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}".format(
        result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9]).split(',')
    return result
    #return [float(i) for i in result]


if __name__ == '__main__':
    import network3
    from network3 import Network
    from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
    from network3 import ReLU
    import os, struct, csv
    from collections import Counter

    expanded_training_data, training_data, validation_data, test_data = \
                    network3.load_data_shared(expanded_time=15)
    
    mini_batch_size = 100
    voters = 1
    vote_box = []
    vote_prob_box = []
    for vote in xrange(voters):
        #expanded_training_data, validation_data, test_data = \
        #                        network3.load_data_shared(expanded_time=10)
        # from book chap6
        #'''
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 29, 29), 
                          filter_shape=(6, 1, 17, 17), poolsize=(1, 1), 
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 6, 13, 13), 
                          filter_shape=(50, 6, 9, 9), poolsize=(1, 1), 
                          activation_fn=ReLU),
            FullyConnectedLayer(
                n_in=50*5*5, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
            FullyConnectedLayer(
                n_in=1000, n_out=500, activation_fn=ReLU, p_dropout=0.5),
            SoftmaxLayer(n_in=500, n_out=10, p_dropout=0.5)], 
            mini_batch_size)

        print "=========Currently Calculating Voter number %s=========" % vote
        k, p = net.SGD(expanded_training_data, 120, mini_batch_size, 0.001, 
                            validation_data, test_data, lmbda=0)
        #vote_box = np.concatenate((vote_box, k))
        #vote_prob_box = np.concatenate((vote_prob_box, np.array(p).reshape(-1,)))
    ### Polishing
    k, p = net.SGD(training_data, 5, mini_batch_size, 0.0005, 
                        validation_data, test_data, lmbda=0)
    vote_box = np.concatenate((vote_box, k))
    vote_prob_box = np.concatenate((vote_prob_box, np.array(p).reshape(-1,)))
    vote_box = vote_box.reshape(voters, -1)
    vote_prob_box = vote_prob_box.reshape(voters, -1, 10)
    l = len(test_data[0].get_value()) #test sample number
    print vote_box.shape
    print vote_prob_box.shape
    print l
    vote_prob_box = np.array([vote_prob_box[j][i] 
                        for i in range(l) for j in range(voters)]).reshape(l, -1, 10)

    print "#Voter = %s, #Testing samples = %s" % (vote_box.shape[0], vote_box.shape[1])


#'''

    voting = []
    for i in xrange(l):
        #vote_result = np.argmax(np.bincount(list(vote_box.T[i])))
        vote_result = vote_box.T[i]
        voting.append(vote_result)

    rows = []
    rows_p = []
    filename="../Desktop/sample.csv"
    with open(filename, 'r') as f:
        f_csv = csv.reader(f)
        for index, row in enumerate(f_csv):
            #if index > 50:
            #    break
            f_name = [row[0]]
            #print voting[index]
            f_pre = chooseVoting(list(voting[index]))
            f_prob = choosePVoting(list(vote_prob_box[index]))
            #pre_digit = voting[index]
            #f_pre = list(k[index])
            #pre_digit = k[index]
            #f_pre = [0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ]
            #f_pre[pre_digit] = 1.0
            new_row = tuple(f_name + f_pre)
            new_row_p = tuple(f_name + list(f_prob))
            rows.append(new_row)
            rows_p.append(new_row_p)
    f.close()

    rows = tuple(rows)
    rows_p = tuple(rows_p)
    print "Start writing csv file..."
    filename="../Desktop/testing%s.csv" %(strftime("%m%d%H%M"))
    with open(filename, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(rows)
    f.close()
    filename="../Desktop/testing%s_p.csv" %(strftime("%m%d%H%M"))
    with open(filename, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(rows_p)
    f.close()
#''' 