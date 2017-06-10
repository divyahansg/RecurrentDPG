# Inspired by: https://github.com/yanpanlau/DDPG-Keras-Torcs/blob/master/ActorNetwork.py

import tensorflow as tf
import keras.backend as K
import numpy as np

# Import various Keras tools.
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Masking, Lambda
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomUniform
from keras.layers.merge import Concatenate

# Actor net.
class Actor:
    def __init__(self, sess, maxSteps, featureDim, actionDim,
                 batchSize, targetUpdate, learningRate,
                 actionScale, actionBias):
        self.sess = sess
        self.batchSize = batchSize
        self.targetUpdate = targetUpdate
        self.actionScale = actionScale
        self.actionBias = actionBias

        # Use existing TF session.
        K.set_session(sess)

        # Create our actor model and an identical target model with its own weights.
        self.model, modelWeights, self.modelHistoryInput = self.createActor(maxSteps, featureDim, actionDim)
        self.target, targetWeights, targetHistoryInput = self.createActor(maxSteps, featureDim, actionDim)

        # For debugging in Jupyter.
        print(self.model.summary())

        # Create graph nodes for computing natural policy gradient.
        self.actionGrads = tf.placeholder(tf.float32, [None, actionDim])
        modelParamsGrads = tf.gradients(self.model.output, modelWeights, -self.actionGrads)

        # Create nodes to apply natural policy gradient to params.
        modelParamsGradsTFStyle = zip(modelParamsGrads, modelWeights)
        AdamOptimizer = tf.train.AdamOptimizer(learningRate)
        self.optimizeModel = AdamOptimizer.apply_gradients(modelParamsGradsTFStyle)

        # Initialize parameters of all models.
        self.sess.run(tf.global_variables_initializer())

    # Run one iteration of Adam on actor model.
    def trainModel(self, histories, actionGrads):
        self.sess.run(self.optimizeModel, feed_dict = {
            self.modelHistoryInput: histories,
            self.actionGrads: actionGrads
        })

    # Soft-update target model.
    def trainTarget(self):
        modelWeights = self.model.get_weights()
        targetWeights = self.target.get_weights()

        # Iteratively update each weight. Note
        # that weights should be in same order.
        for i in range(len(modelWeights)):
            targetWeights[i] = self.targetUpdate * modelWeights[i] + \
                (1 - self.targetUpdate) * targetWeights[i]

        # Set soft-updated weights on target.
        self.target.set_weights(targetWeights)

    # Return a new model instance.
    def createActor(self, maxSteps, featureDim, actionDim):
        historyInput = Input(shape = (maxSteps, featureDim), name = 'History-Input')
        out = Masking(name = 'History-Mask')(historyInput) # Ignore zero timesteps.

        # This part of the model is essentially detachable
        # and can be hot swapped, which is something to add.
        out = LSTM(64)(out)
        out = Dense(128, activation = 'relu')(out)
        # out = BatchNormalization()(out)

        # Last layer must output a valid continuous action.
        init = RandomUniform(minval = -0.003, maxval = 0.003)
        out = Dense(actionDim, activation = 'tanh', kernel_initializer = init)(out)
        out = Lambda(lambda x: x * self.actionScale + self.actionBias)(out)
        model = Model(inputs = historyInput, outputs = out)
        return model, model.trainable_weights, historyInput
