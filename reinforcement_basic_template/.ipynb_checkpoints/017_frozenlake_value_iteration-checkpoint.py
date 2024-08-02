import gymnasium as gym
import numpy as np

map = "4x4"

SLIPPERY = False  # 결정론적 환경
#SLIPPERY = True  # 확률적 환경



# 1. 모든 s에 대해 V(s) = 0으로 초기화


#루프
while True:
    #delta <- 0
    
    #각 s에 대해 반복
    for s in range(num_states):
        # 현재 가치 저장: v <- V(s)
        
        # 업데이트 규칙: V(s) = max_a(sum(p(s,a) * [r + gamma * V(s')]))
        for a in range(num_actions):
            # s', r에 대해 합산 (각 행동 a에 대한 상태 전이 확률과 보상을 반영한 가치 합산)
            for prob, s_, r, _ in transitions[s][a]:
                new_action_values[a] += prob * (r + GAMMA * V[s_])
        
        # delta 갱신: delta <- max(delta, |v_max - V(s)|)
        
        # 상태-가치 함수 업데이트
        
    # delta가 THETA보다 작으면 반복 종료
    

# 액션 가치를 사용해 결정론적 최적 정책 추출
# 결정론적 정책 생성


for s in range(num_states):
    #pi(s) = argmax_a(sum(p(s,a)*[r + gamma*v(s')]))
    

    for a in range(num_actions):
        # s', r에 대해 합산 (각 행동 a에 대한 상태 전이 확률과 보상을 반영한 가치 합산)
        for prob, s_, r, _ in transitions[s][a]:
            action_values[a] += prob * (r + GAMMA * V[s_])

    #pi(s) <- argmax_a(action_values)
    # 각 상태에서 최대 행동 가치를 가지는 행동 선택
    


