# Modified from https://github.com/rlcode/reinforcement-learning-kr by HW Gu

import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gym import wrappers


EPISODES = 300

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = True

        self.state_size = state_size
        self.action_size = action_size

        # hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        # replay memory (max. size = 2000)
        self.memory = deque(maxlen=2000)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_dqn.h5")

    # create an ANN with the state as an input, Q-fun as an output
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # update the target model with the model's weights
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action with the epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save the sample <s, a, r, s'> in replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # learn the model with the batch randomly chosen from the replay memory
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # randomly choose the samples of the batch size from the memory
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # Q-fun of the model for the current state
        # Q-fun of the target net for the next state
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # the target of updating using the Bellman optimality equation
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    env = wrappers.Monitor(env,"./cartpole-experiment/",force=True, video_callable=lambda episode_id: e%10==0)

    state_size = env.observation_space.shape[0] ## 4
    action_size = env.action_space.n            ## 2

    agent = DQNAgent(state_size, action_size)
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        
        # initialize env
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # choose an action from the current state
            action = agent.get_action(state)
            
            # take one step in the env with the chosen action
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # reward of -100 if episode stops midway
            reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r, s'> into the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            
            # learn every timestep
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                # update the target network with the weights of the model at every episode
                agent.update_target_model()
                score = score if score == 500 else score + 100
                
                # print the learning result at every episode
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_dqn.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

                # stop learning if the mean of previous 10 episodes becomes >490
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.model.save_weights("./save_model/cartpole_dqn.h5")
                    sys.exit()
