# TD with linear value function approximation

import model, plots, animate
import numpy as np
import random

sim = model.CartPole()

simtime = 200                                   # duration of episode (seconds)
num_episodes = 50000                            # duration # number of episodes
newtons = 15                                    # force of pushing the cart
box = 1                                         # discretization setting (box 1 = less bins, box 2 = more bins)

# Learning parameters
lr = 1e-3
discount_factor = 0.99
epsilon = 0.5                                   # exploration vs. exploitation

# Network parameters
num_states = 4
num_actions = 2
num_features = 2*num_actions*num_states + 1

score_history = []
best_simtime = 0

# Create linear function approximator
def q_fcn(inputs,weights): # input states, weights vector
    output = np.dot(inputs,weights)
    return output

def he_uniform(num_inputs): # uniform approximator weights initializer
    limit = np.sqrt(6/num_inputs)
    weights = np.random.uniform(-limit,limit,num_inputs)
    return weights
    
def get_sa(states,action): # creates state action pair
    return np.concatenate((states,action))

def get_vfeature(states,action): # creates unique feature vector detailed in report
    if action == 0:
        vfeature = np.concatenate((states,np.square(states)))
        vfeature = np.concatenate((vfeature,np.zeros(2*num_states)))
        vfeature = np.concatenate((vfeature,[1]))
    else:
        vfeature = np.concatenate((np.zeros(2*num_states),states))
        vfeature = np.concatenate((vfeature,np.square(states)))
        vfeature = np.concatenate((vfeature,[1]))
    return vfeature

# Initialize approximator weights
q_weights = he_uniform(num_states+1)
initial_weights = np.copy(q_weights)

for episode in range(num_episodes):

    done = False                                                                # flag to indicate end of episode
    episode_reward = 0                                                          # initialize reward accumulation
    sim = model.CartPole(box=box,force=newtons)                                 # create new cart pole instance
    state = sim.get_cnstates()                                                  # fetch initial state and reshape state vector for Keras network
    if random.uniform(0,1) < epsilon:
        action = random.randint(0,num_actions-1)
    else:
        action = np.argmax([q_fcn(get_sa(state,[0]),q_weights),q_fcn(get_sa(state,[1]),q_weights)])  # get optimal action

    while not done and sim.t < simtime:

        # Take action and advance simulation
        force = sim.discrete_actuator_force(action)

        # Advance simulation
        sim.step(force)

        # Observe resulting state
        nxt_state = sim.get_cnstates()

        # Observe reward and terminal flag
        reward, done = sim.get_cnreward(nxt_state)
        
        if random.uniform(0,1) < epsilon:
            nxt_action = random.randint(0,num_actions-1)
        else:
            nxt_action = np.argmax([q_fcn(get_sa(nxt_state,[0]),q_weights),q_fcn(get_sa(nxt_state,[1]),q_weights)])

        value = q_fcn(get_sa(state,[action]),q_weights)
        nxt_value = q_fcn(get_sa(nxt_state,[nxt_action]),q_weights)

        # Learning algorithm
        delta = reward + discount_factor*nxt_value - value

        # q_weights += delta*lr*get_vfeature(state,action)
        q_weights += delta*lr*get_sa(state,[action])

        episode_reward += reward                                                    # update total reward of episode

        # Update state and action
        state = nxt_state
        action = nxt_action

        # decay epsilon, lr
        # epsilon -= (0.5-1e-5)/num_episodes
        # lr -= (1e-3-1e-6)/num_episodes

    score_history.append(episode_reward)                                            # store reward
    avg_score = np.mean(score_history[-100:])
    if episode % 100 == 0:
        # print('Episode: ',episode,'  ||  Episode reward: ',episode_reward,'  ||  Average score: ',round(avg_score,2),'  ||  Action: ',sim.action_list,'  ||  Weights: ',q_weights)
        print('Episode: ',episode,'  ||  Episode reward: ',episode_reward,'  ||  Average score: ',round(avg_score,2))

    # save best sim
    if sim.t_list[-1] > best_simtime:
        best_simtime = sim.t_list[-1]
        best_sim = sim

    if sim.t >= simtime:
        print('Survival objective reached')
        break

    # break if approximator weights blow up
    if np.isnan(q_weights).any():
        print('NAN error')
        break

print("Survived for "+str(round(best_sim.t_list[-1],2))+'s')
print('Theta variance: '+str(np.rad2deg(np.var(best_sim.theta_list))))
print('X variance: '+str(np.var(best_sim.x_list)))
print('Initial weights: ',initial_weights)
print('Final weights: ',q_weights)

plots.perf_hist(episode+1,score_history)
animate.animate(best_sim)