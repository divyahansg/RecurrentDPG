# Inspired by: https://github.com/yanpanlau/DDPG-Keras-Torcs/blob/master/CriticNetwork.py

import tensorflow as tf
import keras.backend as K
import numpy as np

# Import various Keras tools.
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Masking
from keras.layers.merge import Concatenate, Add
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomUniform
from keras.optimizers import Adam

# Critic net.
class Critic:
    def __init__(self, sess, maxSteps, featureDim, actionDim,
                 batchSize, targetUpdate, learningRate):
        self.sess = sess
        self.batchSize = batchSize
        self.targetUpdate = targetUpdate

        # Use existing TF session.
        K.set_session(sess)

        # Create our critic model and an identical target model with its own weights.
        self.model, self.modelHistoryInput, self.modelActionInput = self.createCritic(maxSteps, featureDim,
                                                                                      actionDim, learningRate)
        self.target, targetHistoryInput, targetActionInput = self.createCritic(maxSteps, featureDim, actionDim,
                                                                               learningRate)

        # For debugging in Jupyter.
        print(self.model.summary())

        # Create a graph node for gradient of model output WRT action input.
        self.modelActionGrads = tf.gradients(self.model.output, self.modelActionInput)

        # Initialize parameters of all models.
        self.sess.run(tf.global_variables_initializer())

    # Compute gradient of critic model output WRT action.
    def modelActionGradients(self, histories, actions):
        results = self.sess.run(self.modelActionGrads, feed_dict = {
            self.modelHistoryInput: histories,
            self.modelActionInput: actions
        })

        # Session run returns a
        # list of output tensors.
        return results[0]

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
    def createCritic(self, maxSteps, featureDim, actionDim, learningRate):
        historyInput = Input(shape = (maxSteps, featureDim), name = 'History-Input')
        historyOut = Masking(name = 'History-Mask')(historyInput) # Ignore zero timesteps.
        actionInput = Input(shape = (actionDim,), name = 'Action-Input')

        # This part of the model is essentially detachable
        # and can be hot swapped, which is something to add.
        historyOut = LSTM(64)(historyOut)
        actionOut = Dense(64)(actionInput)
        # actionOut = BatchNormalization()(actionOut)
        out = Add()([historyOut, actionOut])
        out = Dense(128, activation = 'relu')(out)
        # out = BatchNormalization()(out)

        # Last layer must output a single Q-value.
        init = RandomUniform(minval = -0.003, maxval = 0.003)
        out = Dense(1, activation = 'linear', name = 'Q-Value', kernel_initializer = init)(out)
        model = Model(inputs = [historyInput, actionInput], outputs = out)

        # Critic has a simple MSE loss WRT y.
        AdamOptimizer = Adam(learningRate)
        model.compile(loss = 'mse', optimizer = AdamOptimizer)
        return model, historyInput, actionInput
