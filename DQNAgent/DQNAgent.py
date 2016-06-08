import tensorflow as tf
import numpy as np
import random
from collections import deque
from layers.layers import *
from simple_solver import step

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.95  # decay rate of past observations
OBSERVE = 1000.  # timesteps to observe before training
EXPLORE = 15000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
INITIAL_EPSILON = 0.9  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 4  # size of minibatch

class DQN:
    def __init__(self, actions = 2, input_size=80, pretrained_model=None, update_rule = 'adam', dtype=np.float64):
        self.replayMemory = deque()
        self.timeStep = 0
        self.currentState = None

        self.epsilon = INITIAL_EPSILON
        self.dtype = dtype
        self.conv_params = []
        self.pool_params = []
        self.input_size = input_size
        self.actions = actions
        self.update_rule = update_rule

        # hardcoded conv params
        self.conv_params.append({'stride': 4, 'pad': 2})
        self.conv_params.append({'stride': 2, 'pad': 1})
        self.conv_params.append({'stride': 1, 'pad': 1})

        self.filter_sizes = [8,4,3]
        self.num_filters = [16,32,32]
        self.fullconnected_hidden_dim = 256

        self.pool_params.append({'pool_height':2, 'pool_width':2, 'stride':2})
        self.pool_params.append({'pool_height':3, 'pool_width':3, 'stride':1})
        self.pool_params.append({'pool_height':2, 'pool_width':2, 'stride':1})



        # self.bn_params = []
        cur_size = input_size
        prev_dim = 4
        self.params = {}
        for i, (f, next_dim) in enumerate(zip(self.filter_sizes, self.num_filters)):
            fan_in = f * f * prev_dim
            self.params['W%d' % (i + 1)] = np.sqrt(2.0 / fan_in) * np.random.randn(next_dim, prev_dim, f, f)
            self.params['b%d' % (i + 1)] = np.zeros(next_dim)
            conv_param = self.conv_params[i]
            pool_param = self.pool_params[i]
            cur_size = (cur_size + 2 * conv_param['pad'] - f) / conv_param['stride'] + 1
            cur_size = (cur_size - pool_param['pool_height']) / pool_param['stride'] + 1
            # self.params['gamma%d' % (i + 1)] = np.ones(next_dim)
            # self.params['beta%d' % (i + 1)] = np.zeros(next_dim)
            # self.bn_params.append({'mode': 'train'})
            prev_dim = next_dim

        # Add a fully-connected layers
        fan_in = cur_size * cur_size * self.num_filters[-1]
        self.params['W%d' % (i + 2)] = np.sqrt(2.0 / fan_in) * np.random.randn(fan_in, self.fullconnected_hidden_dim)
        self.params['b%d' % (i + 2)] = np.zeros(self.fullconnected_hidden_dim)
        # self.params['gamma%d' % (i + 2)] = np.ones(hidden_dim)
        # self.params['beta%d' % (i + 2)] = np.zeros(hidden_dim)
        # self.bn_params.append({'mode': 'train'})
        self.params['W%d' % (i + 3)] = np.sqrt(2.0 / self.fullconnected_hidden_dim) * np.random.randn(self.fullconnected_hidden_dim, self.actions)
        self.params['b%d' % (i + 3)] = np.zeros(self.actions)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


    def forward(self, X, start=None, end=None, mode='test'):
        X = X.astype(self.dtype)
        if start is None: start = 0
        if end is None: end = len(self.conv_params) + 1
        layer_caches = []

        prev_a = X
        for i in xrange(start, end + 1):
            i1 = i + 1
            if 0 <= i < len(self.conv_params):
                # This is a conv layer
                w, b = self.params['W%d' % i1], self.params['b%d' % i1]
                # gamma, beta = self.params['gamma%d' % i1], self.params['beta%d' % i1]
                conv_param = self.conv_params[i]
                pool_param = self.pool_params[i]

                # bn_param = self.bn_params[i]
                # bn_param['mode'] = mode
                # next_a, cache = conv_bn_relu_forward(prev_a, w, b, gamma, beta, conv_param, bn_param)
                next_a, cache = conv_relu_pool_forward(prev_a, w, b, conv_param, pool_param)

            elif i == len(self.conv_params):
                # This is the fully-connected hidden layer
                w, b = self.params['W%d' % i1], self.params['b%d' % i1]
                # gamma, beta = self.params['gamma%d' % i1], self.params['beta%d' % i1]
                # bn_param = self.bn_params[i]
                # bn_param['mode'] = mode
                # next_a, cache = affine_bn_relu_forward(prev_a, w, b, gamma, beta, bn_param)
                next_a, cache = affine_relu_forward(prev_a, w, b)
            elif i == len(self.conv_params) + 1:
                # This is the last fully-connected layer that produces scores
                w, b = self.params['W%d' % i1], self.params['b%d' % i1]
                next_a, cache = affine_forward(prev_a, w, b)
            else:
                raise ValueError('Invalid layer index %d' % i)

            layer_caches.append(cache)
            prev_a = next_a

        out = prev_a
        cache = (start, end, layer_caches)
        return out, cache


    def backward(self, dout, cache):
        start, end, layer_caches = cache
        dnext_a = dout
        grads = {}
        for i in reversed(range(start,end + 1)):
            i1 = i + 1
            if i == len(self.conv_params) + 1:
                dprev_a, dw, db = affine_backward(dnext_a, layer_caches.pop())
                grads['W%d' % i1] = dw
                grads['b%d' % i1] = db
            elif i == len(self.conv_params):
                temp = affine_relu_backward(dnext_a, layer_caches.pop())
                dprev_a, dw, db = temp
                grads['W%d' % i1] = dw
                grads['b%d' % i1] = db
                # grads['gamma%d' % i1] = dgamma
                # grads['beta%d' % i1] = dbeta
            elif 0 <= i < len(self.conv_params):
                temp = conv_relu_pool_backward(dnext_a, layer_caches.pop())
                dprev_a, dw, db = temp
                grads['W%d' % i1] = dw
                grads['b%d' % i1] = db
                # grads['gamma%d' % i1] = dgamma
                # grads['beta%d' % i1] = dbeta
            else:
                raise ValueError('Invalid layer index %d' % i)
            dnext_a = dprev_a
        dX = dnext_a
        return dX, grads

    def setPerception(self, nextObservation, action, reward, terminal):
        current = self.currentState.copy()
        current = np.append(current[1:,:],nextObservation).reshape(4,80,80)
        newState = current

        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.train()

        self.currentState = newState
        self.timeStep += 1


    def setInitState(self, observation):
        self.currentState = np.vstack((observation, observation, observation, observation)).reshape(4,80,80)


    def getAction(self):
        QValue = self.loss(self.currentState.reshape(1,4,80,80))
        action = np.zeros(self.actions)
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action


    def loss(self, X, a = None, y=None):

        # Note that we implement this by just caling self.forward and self.backward
        mode = 'test' if y is None and a is None else 'train'
        scores, cache = self.forward(X, mode=mode)
        if mode == 'test':
            return scores
        loss, dscores = mean_square(scores, y, a)
        dX, grads = self.backward(dscores, cache)
        return loss, grads

    def train(self):
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        state_batch = np.asarray(state_batch)
        action_batch = np.asarray(action_batch)
        reward_batch = np.asarray(reward_batch)
        nextState_batch = np.asarray(nextState_batch)

        y_batch = np.zeros(action_batch.shape)
        QValue_batch = self.loss(nextState_batch)
        for i in range(0, BATCH_SIZE):
            action = action_batch[i]
            terminal = minibatch[i][4]
            if terminal:
                y_batch[i] = reward_batch[i]
                y_batch[i] *= action
            else:
                y_batch[i] = (reward_batch[i] + GAMMA * np.max(QValue_batch[i]))
                y_batch[i] *= action

        loss, grads = self.loss(state_batch,a = action_batch,y = y_batch)
        step(self.params,grads,self.update_rule)

        '''
        # save network every 100000 iteration
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '--dqn_agent', global_step=self.timeStep)
        '''


