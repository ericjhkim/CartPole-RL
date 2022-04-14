# Actor critic environment

import model, plots, animate
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

sim = model.CartPole()

simtime = 30                                   # duration of episode (seconds)
num_episodes = 300                             # number of episodes
newtons = 15                                   # force of pushing the cart
box = 1                                        # discretization setting (box 1 = less bins, box 2 = more bins)

# Learning parameters
actor_lr = 1e3
critic_lr = 5e3
discount_factor = 0.99

# Network parameters
num_states = 4
num_actions = 2
num_hidden = 24

# Create network models
actor = Sequential()
actor.add(Dense(num_hidden,input_dim=num_states,activation='relu',kernel_initializer='he_uniform'))
actor.add(Dense(num_actions,activation='softmax',kernel_initializer='he_uniform'))
actor.compile(loss='categorical_crossentropy',optimizer=Adam(lr=actor_lr))

critic = Sequential()
critic.add(Dense(num_hidden,input_dim=num_states,activation='relu',kernel_initializer='he_uniform'))
critic.add(Dense(1,activation='linear',kernel_initializer='he_uniform'))
critic.compile(loss='mse',optimizer=Adam(lr=critic_lr))

score_history = []

best_simtime = 0

for episode in range(num_episodes):

    done = False                                                                # flag to indicate end of episode
    episode_reward = 0
    sim = model.CartPole(box=box,force=newtons)                                 # create new cart pole instance
    state = np.reshape(sim.get_cnstates(), [1, num_states])                     # fetch initial state and reshape state vector for Keras network

    while not done and sim.t < simtime:

        # Sample action from action probability distribution
        action_probs = actor.predict(state,batch_size=1)
        action = np.random.choice(num_actions, p=np.squeeze(action_probs))      # select action based on probabilities

        # Take action and advance simulation
        force = sim.discrete_actuator_force(action)
        sim.step(force)

        # Observe resulting state
        nxt_state = sim.get_cnstates()
        nxt_state = np.reshape(nxt_state, [1, num_states])

        # Observe reward and terminal flag
        reward, done = sim.get_cnreward(nxt_state[0])
        
        # Learning algorithm
        target = np.zeros((1, 1))
        advantages = np.zeros((1, num_actions))
        value = critic.predict(state)[0]
        nxt_value = critic.predict(nxt_state)[0]

        advantages[0][action] = reward + discount_factor*nxt_value*(1-int(done)) - value
        target[0][0] = reward + discount_factor*nxt_value*(1-int(done))

        actor.fit(state, advantages, epochs=1, verbose=0)
        critic.fit(state, target, epochs=1, verbose=0)

        episode_reward += reward                                                # update total reward of episode

        # Update current state
        state = nxt_state

    score_history.append(episode_reward)                                        # store reward
    avg_score = np.mean(score_history[-100:])
    print('Episode: ',episode,'  ||  Episode reward: ',episode_reward,'  ||  Average score: ',round(avg_score,2))

    # save best sim
    if sim.t_list[-1] > best_simtime:
        best_simtime = sim.t_list[-1]
        best_sim = sim

    # end episodes when survived for full simulation time
    if sim.t >= simtime:
        print('Survival objective reached')
        print(np.rad2deg(np.min(best_sim.theta_list)),np.rad2deg(np.max(best_sim.theta_list)),np.min(best_sim.x_list),np.max(best_sim.x_list))
        break

print("Survived for "+str(round(best_sim.t_list[-1],2))+'s')
print('Theta variance: '+str(np.rad2deg(np.var(best_sim.theta_list))))
print('X variance: '+str(np.var(best_sim.x_list)))

plots.perf_hist(episode+1,score_history)
animate.animate(best_sim)