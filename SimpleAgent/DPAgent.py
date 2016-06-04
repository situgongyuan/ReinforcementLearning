from Agent import Agent
import numpy as np
from utils import sample_weighted

class DPAgent(Agent):
    def __init__(self,environment,gamma):
        super(environment)
        self.gamma = gamma
        self.reset()

    """
    initialize the policy to uniform distribution
    """
    def reset(self):
        self.ns = self.env.get_num_states()
        self.na = self.env.get_max_num_actions()
        self.value_table = np.zeros((self.ns,))
        self.policy_table = np.zeros((self.ns,self.na))
        for s in range(self.ns):
            actions = self.env.get_allowed_actions(s)
            self.policy_table[s,actions] = 1.0 / len(actions)

    """
    act according to the policy
    """
    def act(self,s):
        actions = self.env.get_allowed_actions(s)
        ps = self.policy_table[s,actions]
        maxi = sample_weighted(ps)
        return ps[maxi]


    def learn(self):
        self.evaluate_policy()
        self.improve_policy()

    """
    one iteration of policy evaluation of Bellman equation,basically we can run many iterations until converge
    """
    def evaluate_policy(self):
        values = np.zeros((self.ns,))
        for s in range(self.ns):
            expectation_value = 0.0
            actions = self.env.get_allowed_actions(s)
            for a in actions:
                ps = self.policy_table[s,a]
                if ps == 0.0:
                    continue
                else:
                    next_state = self.env.get_next_state(s,a)
                    reward = self.env.get_reward(s,a,next_state)
                    expectation_value += ps * (reward + self.gamma * self.value_table[next_state])
            self.value_table[s] = expectation_value
        self.value_table = values


    def improve_policy(self):
        for s in range(self.ns):
            vmax = -np.inf
            nmax = 0
            vs = []
            actions = self.env.get_allowed_actions(s)
            for a in actions:
                next_state = self.env.get_next_state(s,a)
                reward = self.env.get_reward(s,a,next_state)
                v = reward + self.gamma * self.value_table[next_state]
                vs.append(v)
                if v > vmax:
                    vmax = v
                    nmax = 1
                elif v == vmax:
                    nmax += 1
                else:
                    continue
            vs = np.asarray(vs)
            for i in range(len(vs)):
                self.value_table[s,actions[i]] = 1.0 / nmax if vs[i] == vmax else 0.0







