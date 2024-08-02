import gymnasium as gym
import numpy as np
import pprint

# SFFF       (S: 시작점, 안전)
# FHFH       (F: 얼어있는 표면, 안전)
# FFFH       (H: 구멍, 추락)
# HFFG       (G: 목표, 프리스비 위치)

# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3

map = "4x4"
 
SLIPPERY = False # 결정론적 환경
#SLIPPERY = True  # 확률적 환경

# FrozenLake-v1 환경을 생성합니다.
# desc: None이면 기본 맵을 사용합니다.
# map_name: 사용할 맵의 이름을 지정합니다. 
# is_slippery: True이면 미끄러운 표면(확률적 환경)을 사용하고, False이면 결정론적 환경을 사용합니다.
env = gym.make('FrozenLake-v1', desc=None, map_name=map, is_slippery=SLIPPERY)

GAMMA = 1.0  # 감가율 (discount factor), 미래 보상의 현재 가치
THETA = 1e-5   # 정책 평가의 종료 조건, 변화가 이 값보다 작으면 평가 종료
num_states = env.observation_space.n   # 환경의 상태(state) 개수
num_actions = env.action_space.n    # 환경에서 가능한 행동(action) 개수
transitions = env.P   # 상태-행동 전이 확률 (state-action transition probabilities)
# 환경의 동역학 출력
#pprint.pprint(transitions)

# 평가할 정책인 pi를 입력
# 무작위 정책으로 초기화
policy = np.ones([num_states, num_actions]) * 0.25

# 모든 s에 대해 V(s) = 0 배열을 초기화
V = np.zeros(num_states)

#반복
while True:
    #delta <- 0
    delta = 0
    #각 s에 대해 반복:
    for s in range(num_states):
        # 현재 가치 저장: v <- V(s)
        old_value = V[s]
        new_value = 0
        #업데이트 규칙 : V(s) = sum(pi(a|s)*sum(p(s,a)*[r + gamma*v(s')]))
        for a, prob_action in enumerate(policy[s]):
            # s', r에 대해 합산 (각 행동 a에 대한 상태 전이 확률을 반영한 보상 합산)
            for prob, s_, reward, _ in transitions[s][a]:
                new_value += prob_action * prob * (reward + GAMMA * V[s_])
        # 상태 s의 가치를 새로운 값으로 업데이트
        V[s] = new_value
        # delta 갱신 : delta <- max(delta|v - V(s)|) 
        delta = max(delta, np.abs(old_value - V[s]))

    #delta < theta까지
    if delta < THETA:
        break
#V는 v_pi에 수렴
print("수렴한 Optimal Value = \n", V.reshape(4, 4) if map == "4x4" else V.reshape(8, 8))
print()
print("4x4 map")
print("""
SFFF       
FHFH      
FFFH     
HFFG      
""")