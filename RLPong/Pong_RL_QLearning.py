#PONG ENVIRONMENT
#Action space that is relevant
#0 is Do Nothing  #2 is Go UP  #5 is Go DOWN

#Observation space
#a = Opponent paddle position  632 - 710 
#b = Ball position   0 - 6319
#c = Agent paddle position   5530 - 5608

import gym
import datetime
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

resume = True   #Change from True or False depending on if you have a model

env = gym.make("PongNoFrameskip-v4")  #Creates the Pong environment 
env.reset()
startTime = datetime.datetime.now()


DISCOUNT = 0.995  #measure of how important we find future actions (Weight of future reward vs current reward)
EPISODES = 10000  #number of attempts at a full game
SHOW_EVERY = 100  #when to render the game

actionSpace = np.array([0, 2, 5])  
obsLow = np.array([632, 0, 5530])     #smallest values for observation (pixel values)
obsHigh = np.array([711, 6320, 5609]) #largest values

DISCRETE_OS_SIZE = [5] * len(obsHigh) #5x5x5 combinations of paddle and ball position
discrete_os_win_size = (obsHigh - obsLow) / DISCRETE_OS_SIZE

epsilon = 0.50 # chance to perform a random, exploratory action 
START_EPSILON_DECAYING = 1  #over time we want our model to stop exploring
END_EPSILON_DECAYING = EPISODES // 2  #always divide out to integer
epislon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=0, high=0, size= (DISCRETE_OS_SIZE + [np.size(actionSpace)]))
#table of all possible combinations of position and velocity plus 3 possible actions for every combination

if resume:
  model = pickle.load(open('pong3v4Winning.pickle', 'rb'))
  q_table = model['Q']
  epsilon = model['Epsilon']
  total_episodes = model['Episode']
  aggr_ep_rewards = model['Stats']
else:
  model = {}
  model['Q'] = q_table
  model['Epsilon'] = epsilon
  model['Episode'] = 0
  total_episodes = 0
  aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

ep_rewards = []

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
  I = I[35:193] # crop - remove 35px from start & 16px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
  I = I[::2,::2,0] # downsample by factor of 2.
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I = I.astype(np.float).transpose().ravel() # ravel flattens an array and collapses it into a column vector
  a = np.where(I==213)  #213 is opponent paddle index from 632 to 710
  if np.size(a) ==0: #if no paddle yet initalize at obsLow
      a = obsLow[0] + ((obsHigh[0] - obsLow[0]) // 2)
  elif a[0][0] >= obsHigh[0]: #check if pong glitched and shows out of bounds
      a = obsHigh[0] - 1
  elif a[0][0] < obsLow[0]:
      a = obsLow[0]
  else:
      a = a[0][0]
  b = np.where(I==236)  #236 is ball index from 0 to 6319
  if np.size(b) ==0: #if no paddle yet initalize at obsLow
      b = obsLow[1] + ((obsHigh[1] - obsLow[1]) // 2)
  elif b[0][0] >= obsHigh[1]: #check if pong glitched and shows out of bounds
      b = obsHigh[1] - 1
  elif b[0][0] < obsLow[1]:
      b = obsLow[1]
  else:
      b = b[0][0]
  c = np.where(I==92)  #92 is your paddle index from 5530 to 5608
  if np.size(c) ==0: #if no paddle yet initalize at obsLow
      c = obsLow[2]
  elif c[0][0] >= obsHigh[2]: #check if pong glitched and shows out of bounds
      c = obsHigh[2] - 1
  elif c[0][0] < obsLow[2]:
      c = obsLow[2]
  else:
      c = c[0][0]
  obsv = np.array([a,b,c])
  return obsv

def learning_rate(n : int , min_rate=0.01 ) -> float  :
    #Decaying learning rate 
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

def get_discrete_state(state):
	discrete_state = (state - obsLow) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
	episode_reward = 0
	total_episodes = total_episodes + 1
	LEARNING_RATE = learning_rate(total_episodes)
	if episode % SHOW_EVERY == 0:
		render = True
	else:
		print('Episode: ', total_episodes)
		render = False
    
	observation = env.reset()    #get initial state
	obsv = prepro(observation)   #apply preprocessing 
	discrete_state = get_discrete_state(obsv)
	done = False
    
	while not done:

		if np.random.random() > epsilon:
			action = actionSpace[np.argmax(q_table[discrete_state])] #exploitation
		else:
			action = np.random.choice(actionSpace) #exploration 

		actionIndex = np.where(actionSpace == action)[0][0]
		new_state, reward, done, info = env.step(action)
		episode_reward += reward
		new_obsv = prepro(new_state)
		new_discrete_state = get_discrete_state(new_obsv)
        
		if render:
			env.render()

		if not done:

			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (actionIndex, )]
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state + (actionIndex, )] = new_q

		discrete_state = new_discrete_state

	if END_EPSILON_DECAYING >= total_episodes >= START_EPSILON_DECAYING:
		epsilon -= epislon_decay_value

	ep_rewards.append(episode_reward)

	if not episode % SHOW_EVERY:
		model['Q'] = q_table
		model['Epsilon'] = epsilon
		model['Episode'] = total_episodes
		average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
		aggr_ep_rewards['ep'].append(total_episodes)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
		aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
		model['Stats'] = aggr_ep_rewards
		pickle.dump(model, open('pong3v4Winning.pickle', 'wb'))
        
		TimePassed = datetime.datetime.now() - startTime
		print("Total time passed - ", TimePassed)
		print(f"Episode: {total_episodes} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")

env.close()

print(aggr_ep_rewards['ep'])

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()