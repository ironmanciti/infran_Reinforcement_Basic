import gymnasium as gym
import numpy as np

# SFFF       (S: 시작점, 안전)
# FHFH       (F: 얼어있는 표면, 안전)
# FFFH       (H: 구멍, 추락)
# HFFG       (G: 목표, 프리스비 위치)

map = "4x4"

SLIPPERY = False  # 결정론적 환경
#SLIPPERY = True  # 확률적 환경



# 1. 모든 s에 대해 V(s) = 0과 임의의 pi(s)를 초기화


while not policy_stable:
    #2. 정책 평가
    while True:
        #delta <- 0
        
        #각 s에 대해 반복
        for s in range(num_states):
            
            #V(s) = sum(pi(a|s)*sum(p(s,a)*[r + gamma*v(s')]))
            for a, prob_a in enumerate(pi[s]):
                # s', r에 대해 합산
                for prob, s_, r, _ in transitions[s][a]:
                    new_value += prob_a * prob * (r + GAMMA * V[s_])
            
            #delta <-max(delta|v - V(s)|)
            
        #delta < theta까지
        

    #3. 정책 개선
    #policy_stable <- true
    
    #각 s에 대해:
    for s in range(num_states):
        # pi_s <- argmax_a(sum(p(s',r|s,a)*[r + gamma*V(s')]))
        
        for a in range(num_actions):
            for prob, s_, r, _ in transitions[s][a]:
                new_action_values[a] += prob * (r + GAMMA * V[s_])



    #정책이 안정화 되면 V와 pi를 반환하고 종료, 아니면 2로 돌아감.


