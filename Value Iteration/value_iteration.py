import gymnasium as gym
import random

env = gym.make('CliffWalking-v0', is_slippery=True, render_mode='human').unwrapped

action_space = env.action_space
state_space = env.observation_space

v = [0 for _ in range(state_space.n)]

sigma = 0.0001
gamma = 0.9
delta = -1

def value_iteratiion_is_finished(delta,sigma):
    return delta != -1 and delta < sigma

while not value_iteratiion_is_finished(delta,sigma):
    delta = 0
    for s in range(state_space.n):
        v_old = v[s]
        v[s] = max(sum(prob * (reward + gamma * v[nex_state]) if nex_state!=47 else prob * (100 + gamma * v[nex_state])
                        for prob, nex_state, reward, terminated in env.P[s][a])
                    for a in range(action_space.n))
        delta = max(delta, abs(v_old - v[s]))

policy = {}

for s in range(state_space.n):
    policy[s] = max(range(action_space.n), key=lambda a: sum(prob * (reward + gamma * v[next_state]) if next_state != 47 else prob * (100 + gamma * v[next_state])
                                                            for prob, next_state, reward, terminated in env.P[s][a]))

grid = [[' ' for _ in range(12)] for _ in range(4)]
for state in range(state_space.n):
    row = state // 12
    col = state % 12
    if state == 36:
        grid[row][col] = 'G'
    elif state == 47:
        grid[row][col] = 'C'
    else:
        grid[row][col] = ['^', '>', 'v', '<'][policy[state]]

for row in grid:
    print(row)

test_episodes = 5
for i in range(test_episodes):
    state, info = env.reset()
    done = False
    while not done:
        action = policy[state]
        state, reward, done, truncated, info = env.step(action)