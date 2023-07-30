############################################################
#                                                          #
#   Daniel Keitley, Ngoc Khanh Nguyen - Deep Learning CW1  #
#                                                          #
############################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow as tf
import cPickle as pickle
import numpy as np
from tensorflow.python import debug as tf_debug


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', os.getcwd() + '/dataset/',
                           'Directory where the dataset will be stored and checkpoint.')
tf.app.flags.DEFINE_integer('max_steps', 1000,
                            'Number of mini-batches to train on.')
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            'Number of steps between logging results to the console and saving summaries')
tf.app.flags.DEFINE_integer('save_model', 1000,
                            'Number of steps between model saves')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch_size', 100, 'Number of examples per mini-batch')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, 'Weight Decay.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Number of examples to run.')
tf.app.flags.DEFINE_integer('img_width', 32, 'Image width')
tf.app.flags.DEFINE_integer('img_height', 32, 'Image height')
tf.app.flags.DEFINE_integer('img_channels', 3, 'Image channels')
tf.app.flags.DEFINE_integer('num_classes', 43, 'Number of classes')
tf.app.flags.DEFINE_string('train_dir',
                           '{cwd}/logs/exp_bs_{bs}'.format(cwd=os.getcwd(),
                                                                   bs=FLAGS.batch_size),
                           'Directory where to write event logs and checkpoint.')

DEBUG = False

def deepnn(x):
    
    #whitening
    with tf.variable_scope('Pre_Process'):
        x_mean, x_var = tf.nn.moments(x, axes=[1],keep_dims=True)
        x = tf.subtract(x,x_mean)
        x = tf.div(x,tf.sqrt(x_var))

    x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])

    img_summary = tf.summary.image('Input_images', x_image)

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.variable_scope('Conv_1'):
        W_conv1 = weight_variable([5, 5, FLAGS.img_channels, 32])
        tf.add_to_collection('decay_weights',W_conv1)
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1,2) + b_conv1)
        h_pool1 = avg_pool_3x3(h_conv1)

    with tf.variable_scope('Conv_2'):
        # Second convolutional layer -- maps 32 feature maps to 32.
        W_conv2 = weight_variable([5, 5, 32, 32])
        tf.add_to_collection('decay_weights',W_conv2)
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,2) + b_conv2)
        h_pool2 = avg_pool_3x3(h_conv2)
	
    with tf.variable_scope('Conv_3'):
        W_conv3 = weight_variable([5, 5, 32, 64])
        tf.add_to_collection('decay_weights',W_conv3)
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3,2) + b_conv3)
        h_pool3 = max_pool_3x3(h_conv3)

    with tf.variable_scope('FC_1'):
        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28
        # image is down to 8x8x64 feature maps -- maps this to 1024 features.
        h_pool3_flat = tf.reshape(h_pool3,[-1,4*4*64])
        W_fc1 = weight_variable([4*4*64,64])
        tf.add_to_collection('decay_weights',W_fc1)
        b_fc1 = bias_variable([64])
        h_fc1 = tf.matmul(h_pool3_flat,W_fc1) + b_fc1

    with tf.variable_scope('FC_2'):
        # Map the 1024 features to 10 classes, one for each digit
        fc1_flat = tf.reshape(h_fc1,[-1,64])
        W_fc2 = weight_variable([64, FLAGS.num_classes])
        tf.add_to_collection('decay_weights',W_fc2)
        b_fc2 = bias_variable([FLAGS.num_classes])
        y_conv = tf.matmul(fc1_flat, W_fc2) + b_fc2

        return y_conv, img_summary


def conv2d(x, W,p):
    '''conv2d returns a 2d convolution layer with full stride.''' 
    output = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name='convolution')
    return tf.pad(output, tf.constant([[0,0],[p, p,],[p, p],[0,0]]), "CONSTANT")


def avg_pool_3x3(x):
    '''avg_pool_3x3 downsamples a feature map by 3X.'''
    output = tf.nn.avg_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='VALID', name='pooling')
    # think this is right...
    return tf.pad(output, tf.constant([[0,0],[0, 1,],[0, 1],[0,0]]), "CONSTANT")

def max_pool_3x3(x):
    '''max_pool_3x3 downsamples a feature mah_pool1p by 3X.'''
    output = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='VALID', name='pooling2')
    return tf.pad(output, tf.constant([[0,0],[0, 1], [0, 1],[0,0]]), "CONSTANT")

def weight_variable(shape):
    '''weight_variable generates a weight variable of a given shape.'''
    initial = tf.random_uniform(shape, -0.05,0.05)
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    '''bias_variable generates a bias variable of a given shape.'''
    initial = tf.random_uniform(shape, -0.05,0.05)
    #initial = tf.zeros((shape), dtype = tf.float32)
    return tf.Variable(initial, name='biases')


def batch_generator(dataset, group, batch_size=FLAGS.batch_size):
	idx = 0
	dataset = dataset[0] if group == 'train' else dataset[1]

	dataset_size = len(dataset)
	indices = range(dataset_size)
	np.random.shuffle(indices)
	while idx < dataset_size:
		chunk = slice(idx, idx+batch_size)
		chunk = indices[chunk]
		chunk = sorted(chunk)
		idx = idx + batch_size
		yield [dataset[i][0] for i in chunk], [dataset[i][1] for i in chunk]


def main(_):
    tf.reset_default_graph()

    dataset = pickle.load(open('dataset.pkl', 'rb'))
    learning_rate = 0.01
    current_validation_acc = 1

    with tf.variable_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    # Build the graph for the deep net
    y_conv, img_summary = deepnn(x)


    # weight decay
    with tf.variable_scope('weights_norm') as scope:
        weights_norm = tf.reduce_sum(
        input_tensor = FLAGS.weight_decay*tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection('decay_weights')]), name='weights_norm')

    tf.add_to_collection('losses', weights_norm)
    with tf.variable_scope('x_entropy'):
        x_entropy = tf.reduce_mean(tf.negative(tf.log(tf.reduce_sum(tf.multiply(tf.nn.softmax(y_conv),y_),1))))
        
		
    tf.add_to_collection('losses', x_entropy)
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    train_step = tf.train.MomentumOptimizer(learning_rate,FLAGS.momentum).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    loss_summary = tf.summary.scalar('Loss', loss)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)

    # summaries for TensorBoard visualisation
    training_summary = tf.summary.merge([img_summary, loss_summary])
    test_summary = tf.summary.merge([img_summary, acc_summary])

    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:
        if(DEBUG):
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(FLAGS.train_dir + '_validate', sess.graph)
        summary_writer_test = tf.summary.FileWriter(FLAGS.train_dir + '_test', sess.graph)

        sess.run(tf.global_variables_initializer())
        step = 0

        # Training and validation
        for (trainImages, trainLabels) in batch_generator(dataset, 'train'):

            step = step + 1
            print(step)
            if(step >= FLAGS.max_steps):
                break

            if(step % 45 ==0):
                learning_rate = learning_rate/10

            trainImages = np.reshape(trainImages,[-1, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
            _, summary_str = sess.run([train_step, training_summary], feed_dict={x: trainImages, y_: trainLabels})


            # Validation - for testing purposes only
            #(testImages,testLabels) = next(batch_generator(dataset,'test'))
            #testImages = np.reshape(testImages,[-1, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
            #validation_summary = tf.summary.merge([loss_summary, acc_summary])
            #validation_accuracy, summary_str = sess.run([accuracy,validation_summary],feed_dict={x: testImages, y_: testLabels})
            #print('step {}, accuracy on validation set : {}'.format(step, validation_accuracy))
            #summary_writer_validation.add_summary(summary_str, step)


            if step % (FLAGS.log_frequency + 1)== 0:
                summary_writer.add_summary(summary_str, step)
            
            # Save the model checkpoint periodically.
            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir + '_train', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        
        # Testing
        # resetting the internal batch indexes
        evaluatedImages = 0
        test_accuracy = 0
        nRuns = 0

        for (testImages, testLabels) in batch_generator(dataset, 'test'):
            # don't loop back when we reach the end of the test set
            testImages = np.reshape(testImages,[-1, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
            test_accuracy_temp, _ = sess.run([accuracy, test_summary], feed_dict={x: testImages, y_: testLabels})
            nRuns = nRuns + 1
            test_accuracy = test_accuracy + test_accuracy_temp
            evaluatedImages = evaluatedImages + np.array(testLabels).shape[0]
        test_accuracy = test_accuracy / nRuns
        print('test set: accuracy on test set: %0.3f' % test_accuracy)


if __name__ == '__main__':
    tf.app.run(main=main)