from Agent import Agent
import numpy as np
from utils import sample_weighted

class TDAgent(Agent):
    def __init__(self,environment,parameters):
        self.env = environment
        self.parameters = self.get_default_parameters() if parameters is None else parameters
        self.reset()

    def reset(self):
        self.ns = self.env.get_num_states()
        self.na = self.env.get_max_num_actions()
        self.state_action_values = np.zeros((self.ns,self.na)) if self.parameters['q_values'] is None else self.parameters['q_values']
        self.policy_table = np.zeros((self.ns,self.na))
        self.eligible_trace_table = np.zeros((self.ns,self.na))
        for s in range(self.ns):
            actions = self.env.get_allowed_actions(s)
            self.policy_table[s,actions] = 1.0 / len(actions)
        self.s0 = None
        self.a0 = None
        self.r0 = None
        self.s1 = None
        self.a1 = None

    """
    act according to epsilon-greedy policy
    """
    def act(self,s):
        actions = self.env.get_allowed_actions(s)
        ps = self.policy_table[s,actions]
        if (np.random.rand() < self.parameters['epsilon']):
            a = actions[np.random.randint(0,len(ps))]
        else:
            maxi = sample_weighted(ps)
            a = ps[maxi]
        self.s0 = self.s1
        self.a0 = self.a1
        self.s1 = s
        self.a1 = a
        return a

    def learn(self,r1):
        if not self.r0 is None:
            self.learn_from_transition(self.s0,self.a0,self.r0,self.s1,self.a1,self.parameters['lambda'])
            if (self.parameters['plan'] > 0):
                self.update_model(self.s0,self.a0,self.r0,self.s1,self.a1)
                self.plan()
        self.r0 = r1


    def learn_from_transition(self,s0,a0,r0,s1,a1,lmbd):
        if self.parameters['update'] == 'Q-learning':
            actions = self.env.get_allowed_actions(s1)
            qmax = -np.inf
            for a in actions:
                if (self.state_action_values[s1,a] > qmax):
                    qmax = self.state_action_values[s1,a]
            target = r0 + self.parameters['gamma'] * qmax
        elif self.parameters['update'] == 'SARSA':
            actions = self.env.get_allowed_actions(s1)
            target = r0 + self.parameters['gamma'] * self.state_action_values[s1,a1]

        #TD(lambda) with eligible trace
        if (lmbd > 0):
            if self.parameters['replacing_traces'] == True:
                self.eligible_trace_table[s0,a0] = 1
            else:
                self.eligible_trace_table[s0,a0] += 1
            decay = lmbd * self.parameters['gamma']










