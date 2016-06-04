from Agent import Agent
import numpy as np
from utils import sample_weighted

class TDAgent(Agent):
    def __init__(self,environment,parameters):
        self.env = environment
        self.parameters = self.get_default_parameters() if parameters is None else parameters
        self.reset()

    def get_default_parameters(self):
        return {'update':'Q-learning','plan':0,'Q_values':None,'epsilon':0.1,'gamma':0.75,'lambda':0,'alpha':0.01,'beta':0.01,
                'replacing_traces':True,'explore':False}


    def reset(self):
        self.ns = self.env.get_num_states()
        self.na = self.env.get_max_num_actions()
        self.state_action_values = np.zeros((self.ns,self.na)) if self.parameters['Q_values'] is None else self.parameters['Q_values']
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
        self.env_model_state = np.zeros((self.ns,self.na))
        self.env_model_reward = np.zeros((self.ns,self.na))
        self.priority_queue = np.zeros((self.ns,self.na))
        self.seen_state_action = []


    """
    act according to epsilon-greedy policy
    """
    def act(self,s):
        actions = self.env.get_allowed_actions(s)
        ps = self.policy_table[s,actions]
        if (np.random.rand() < self.parameters['epsilon']):
            a = actions[np.random.randint(0,len(ps))]
            self.parameters['explore'] = True
        else:
            maxi = sample_weighted(ps)
            a = ps[maxi]
            self.parameters['explore'] = False
        self.s0 = self.s1
        self.a0 = self.a1
        self.s1 = s
        self.a1 = a
        return a

    def learn(self,r1):
        if not self.r0 is None:
            self.learn_from_transition(self.s0,self.a0,self.r0,self.s1,self.a1,self.parameters['lambda'])
            if (self.parameters['plan'] > 0):
                self.update_model(self.s0,self.a0,self.r0,self.s1)
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
            state_update = np.zeros((self.ns,))
            for s in range(self.ns):
                actions = self.env.get_allowed_actions(s)
                for a in actions:
                    esa = self.eligible_trace_table[s,a]
                    update = self.parameters['alpha'] * esa * (target - self.state_action_values[s,a])
                    self.state_action_values[s,a] += update
                    self.eligible_trace_table[s,a] *= decay
                    self.update_priority(s,update)
                    u = np.abs(update)
                    if (u > state_update[s]):
                        state_update[s] = u
            for s in range(self.ns):
                if (state_update[s] > 1e-5):
                    self.update_policy(s)
            if (self.parameters['explore'] == True and self.parameters['update'] == 'Q-learning'):
                self.eligible_trace_table = np.zeros((self.ns,self.na))

        #simple update Q(s0,a0)
        else:
            update = self.parameters['alpha'] * (target - self.state_action_values[s0,a0])
            self.state_action_values[s0,a0] += update
            self.update_priority(s0,update)
            self.update_policy(s0)

    def update_policy(self,s):
        actions = self.env.get_allowed_actions(s)
        qs = self.state_action_values[s,actions]
        qmax = np.max(qs)
        is_max = (qs == qmax)
        nmax = np.sum(is_max)
        if self.parameters['smooth_update_policy'] == True:
            #smooth update
            qs[is_max] += self.parameters['beta'] * (1.0 / nmax -  qs[is_max])
            qs[-is_max] += self.parameters['beta'] * (0.0 - qs[-is_max])
        else:
            #hard assign
            qs[is_max] = 1.0 / nmax
            qs[-is_max] = 0.0
        if self.parameters['smooth_update_policy'] == True:
            #normalization
            qs /= np.sum(qs)
        """
        #naive implementation with for loop
        qmax = -np.inf
        nmax = 0
        qs = []
        actions = self.env.get_allowed_actions(s)
        for a in actions:
            q = self.state_action_values[s,a]
            qs.append(a)
            if q > qmax:
                qmax = q
                nmax = 1
            elif q == qmax:
                nmax += 1
            else:
                continue
        qs = np.asarray(qs)
        sum = 0
        for i in range(len(qs)):
            q = qs[i]
            a = actions[i]
            target = 1.0 / nmax if q == qmax else 0.0
            if self.parameters['smooth_update_policy'] == True:
                self.policy_table[s,a] += self.parameters['beta'] * (target - self.policy_table[s,a])
                sum += self.policy_table[s,a]
            else:
                self.policy_table[s,a] = target
        if self.parameters['smooth_update_policy'] == True:
            self.policy_table[s,actions] /= sum
        """

    def update_priority(self,s,update):
        u = np.abs(update)
        if (u < 1e-5 or self.parameters['plan'] == 0):
            return
        for state in range(self.ns):
            actions = self.env.get_allowed_actions(s)
            lead_to_s = (self.env_model_state[state,actions] == s)
            self.priority_queue[state,lead_to_s] += u
            """
                # naive implementation using for loop
                for action in actions:
                if (self.env_model_state[state,action] == s):
                    self.priority_queue[state,action] += u
            """
    def update_model(self,s0,a0,r0,s1):
        # transition (s0,a0) -> (r0,s1) was observed. Update environment model
        if (self.env_model_state[s0,a0] == -1):
            self.seen_state_action.add((s0,a0))
        self.env_model_state[s0,a0] = s1
        self.env_model_reward[s0,a0] = r0

    def plan(self):
        spq = {}
        for (state,action) in self.seen_state_action:
            sap = self.priority_queue[state,action]
            if sap > 1e-5:
                spq[(state,action)] = sap
        spq = sorted(spq.items(),cmp = lambda x,y : -cmp(x[1],y[1]))
        plan_nsteps = np.min(self.parameters['plan'],len(spq))
        for i in range(plan_nsteps):
            (s0,a0) = spq[i][0]
            self.priority_queue[s0,a0] = 0.0
            r0 = self.env_model_reward[s0,a0]
            s1 = self.env_model_state[s0,a0]
            a1 = -1
            if self.parameters['update'] == 'SARSA':
                actions = self.env.get_allowed_actions(s1)
                a1 = actions[np.random.randint(0,len(actions))]
            self.learn_from_transition(s0,a0,r0,s1,a1,0)




















