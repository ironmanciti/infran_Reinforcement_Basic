import gymnasium as gym
import numpy as np

# SFFF       (S: 시작점, 안전)
# FHFH       (F: 얼어있는 표면, 안전)
# FFFH       (H: 구멍, 추락)
# HFFG       (G: 목표, 프리스비 위치)

map = "4x4"

SLIPPERY = False  # 결정론적 환경
#SLIPPERY = True  # 확률적 환경

env = gym.make('FrozenLake-v1', desc=None,
               map_name=map, is_slippery=SLIPPERY)

GAMMA = 1.0
THETA = 1e-7
num_states = env.observation_space.n
num_actions = env.action_space.n
transitions = env.P

# 1. 모든 s에 대해 V(s) = 0과 임의의 pi(s)를 초기화
V = np.zeros(num_states)
pi = np.ones([num_states, num_actions]) * 0.25

policy_stable = False

while not policy_stable:
    #2. 정책 평가
    while True:
        #delta <- 0
        delta = 0
        #각 s에 대해 반복
        for s in range(num_states):
            old_value = V[s]
            new_value = 0
            #V(s) = sum(pi(a|s)*sum(p(s,a)*[r + gamma*v(s')]))
            for a, prob_a in enumerate(pi[s]):
                # s', r에 대해 합산
                for prob, s_, r, _ in transitions[s][a]:
                    new_value += prob_a * prob * (r + GAMMA * V[s_])
            V[s] = new_value
            #delta <-max(delta|v - V(s)|)
            delta = max(delta, np.abs(old_value - V[s]))
        #delta < theta까지
        if delta < THETA:
            break

    #3. 정책 개선
    #policy_stable <- true
    policy_stable = True
    old_pi = pi
    #각 s에 대해:
    for s in range(num_states):
        # pi_s <- argmax_a(sum(p(s',r|s,a)*[r + gamma*V(s')]))
        new_action_values = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, s_, r, _ in transitions[s][a]:
                new_action_values[a] += prob * (r + GAMMA * V[s_])

        new_action = np.argmax(new_action_values)
        pi[s] = np.eye(num_actions)[new_action]

    if old_pi.all() != pi.all():
        policy_stable = False
    #정책이 안정화 되면 V와 pi를 반환하고 종료, 아니면 2로 돌아감.

print("Optimal Value Function = \n", V.reshape(
    4, 4) if map == "4x4" else V.reshape(8, 8))
print("Optimal Policy = \n", pi)
print("Optimal Action = \n", np.argmax(pi, axis=1).reshape(4, 4) if map == "4x4"
      else np.argmax(pi, axis=1).reshape(8, 8))
