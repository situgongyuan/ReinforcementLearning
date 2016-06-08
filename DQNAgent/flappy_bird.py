import cv2
import sys

sys.path.append('/Users/stgy/PyWorkspace/DRL-FlappyBird/game/')
import wrapped_flappy_bird as game
from DQNAgent import DQN
import numpy as np


# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    observation = np.asarray(observation,dtype=np.float64)
    """
    m = np.max(observation)
    observation = 1.0 / m * observation
    """
    return np.reshape(observation, (80, 80))

def play():
    actions = 2
    agent = DQN(actions)
    flappyBird = game.GameState()
    # play game
    # obtain init state
    action0 = np.array([1, 0])
    observation0, reward0, is_terminal = flappyBird.frame_step(action0)

    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    observation0 = np.asarray(observation0,dtype=np.float64)
    agent.setInitState(observation0)

    """
    m = np.max(observation0)
    observation0 = 1.0 / m * observation0
    agent.setInitState(observation0)
    """

    while 1 != 0:
        action = agent.getAction()
        print agent.timeStep
        nextObservation, reward, terminal = flappyBird.frame_step(action)
        nextObservation = preprocess(nextObservation)
        agent.setPerception(nextObservation, action, reward, terminal)


def main():
    play()


if __name__ == '__main__':
    main()
