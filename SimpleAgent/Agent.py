"""
Agent:
"""
class Agent(object):
    def __init__(self,environment):
        self.env = environment

    """
    reset the agent
    """
    def reset(self):
        pass

    """
    act at some state
    """
    def act(self,state):
        pass

    """
    learn from the feedback of the environment
    """
    def learn(self,reward):
        pass

