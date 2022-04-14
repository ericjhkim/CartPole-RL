# TD with nonlinear neural network-based value function approximation

import model, plots, animate
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

sim = model.CartPole()

simtime = 200                                    # duration of episode (seconds)
num_episodes = 300                               # number of episodes
newtons = 15                                     # force of pushing the cart
box = 1                                          # discretization setting (box 1 = less bins, box 2 = more bins)

# Learning parameters
lr = 1e-2
discount_factor = 0.99

# Network parameters
num_states = 4
num_actions = 2
num_hidden = 24

# Create network model
q_net = Sequential()
q_net.add(Dense(num_hidden,input_dim=num_states+1,activation='relu',kernel_initializer='he_uniform'))
q_net.add(Dense(1,activation='linear',kernel_initializer='he_uniform'))
q_net.compile(loss='mse',optimizer=Adam(lr=lr))

score_history = []
best_simtime = 0

def get_action(q_net,state):
    sa0 = create_sa_pair(state,0)
    sa1 = create_sa_pair(state,1)
    val = q_net.predict(sa0)
    action = np.argmax([q_net.predict(sa0)[0],q_net.predict(sa1)[0]])
    return action

def create_sa_pair(states,action):
    sa_pair = [i for i in states[0]]
    sa_pair.append(action)
    sa_pair = np.expand_dims(sa_pair,axis=0)
    return sa_pair

for episode in range(num_episodes):

    done = False                                                                # flag to indicate end of episode
    episode_reward = 0                                                          # initialize reward accumulation
    sim = model.CartPole(box=box,force=newtons)                                 # create new cart pole instance
    state = np.reshape(sim.get_cnstates(), [1, num_states])                     # fetch initial state and reshape state vector for Keras network
    action = get_action(q_net,state)                                            # get optimal action

    while not done and sim.t < simtime:

        # Take action and advance simulation
        force = sim.discrete_actuator_force(action)

        # Advance simulation
        sim.step(force)

        # Observe resulting state
        nxt_state = np.reshape(sim.get_cnstates(), [1, num_states])

        # Observe reward and terminal flag
        reward, done = sim.get_cnreward(nxt_state[0])

        reward = reward if not done or episode_reward == 499 else -100

        nxt_action = get_action(q_net,nxt_state)

        # Learning algorithm
        value = q_net.predict(create_sa_pair(state,action))[0]
        nxt_value = q_net.predict(create_sa_pair(nxt_state,nxt_action))[0]
        
        advantages = np.zeros((1, 2))
        advantages[0][action] = reward + discount_factor*nxt_value - value
        
        q_net.fit(create_sa_pair(state,action), advantages, epochs=1, verbose=0)

        episode_reward += reward                                                    # update total reward of episode

        # Update state and action
        state = nxt_state
        action = nxt_action

    episode_reward = episode_reward if episode_reward == 500.0 else episode_reward + 100
    score_history.append(episode_reward)                                            # store reward
    avg_score = np.mean(score_history[-100:])
    print('Episode: ',episode,'  ||  Episode reward: ',episode_reward,'  ||  Average score: ',round(avg_score,2))

    # save best sim
    if sim.t_list[-1] > best_simtime:
        best_simtime = sim.t_list[-1]
        best_sim = sim

    if sim.t >= simtime:
        print('Survival objective reached')
        break

print("Survived for "+str(round(best_sim.t_list[-1],2))+'s')
print('Theta variance: '+str(np.rad2deg(np.var(best_sim.theta_list))))
print('X variance: '+str(np.var(best_sim.x_list)))

plots.perf_hist(episode+1,score_history)
animate.animate(best_sim)