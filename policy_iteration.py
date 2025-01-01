import gymnasium as gym
import random

env = gym.make('CliffWalking-v0', is_slippery=True, render_mode='human').unwrapped

action_space = env.action_space
state_space = env.observation_space

def evaluation_is_finished(delta, sigma):
    return delta != -1 and delta < sigma

def evaluate_policy(env, policy, v, sigma, gamma):
    delta = -1
    while not evaluation_is_finished(delta,sigma):
        delta = 0
        for state in range(state_space.n):
            v_old = v[state]
            action = policy[state]
            transitions = env.P[state][action]
            v[state] = sum(prob * (reward + gamma * v[next_state]) if next_state!=47 else prob * (100 + gamma * v[next_state]) 
                          for prob, next_state, reward, terminated in transitions)
            delta = max(delta, abs(v_old - v[state]))

def improve_policy(env, policy, v, gamma):
    policy_stable = True
    state = 0
    while state < state_space.n and policy_stable:
        old_action = policy[state]
        max_v_value = -float('inf')
        best_action = -1
        for action in range(action_space.n):
            transitions = env.P[state][action]
            v_value = sum(prob * (reward + gamma * v[next_state]) if next_state!=47 else prob * (100 + gamma * v[next_state]) 
                          for prob, next_state, reward, terminated in transitions)
            if v_value > max_v_value:
                max_v_value = v_value
                best_action = action
        policy[state] = best_action
        if old_action != best_action:
            policy_stable = False
        state += 1
    return policy_stable

policy = {}
v = {}

for state in range(state_space.n):
    policy[state] = random.choice(range(action_space.n))

for state in range(state_space.n):
    v[state] = 0

gamma = 0.99
sigma = 0.01

policy_stable = False
while not policy_stable:
    print('Policy evaluation')
    evaluate_policy(env, policy, v, sigma, gamma)
    print('Policy improvement')
    policy_stable = improve_policy(env, policy, v, gamma)

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