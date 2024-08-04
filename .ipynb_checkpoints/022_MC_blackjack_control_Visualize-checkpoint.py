# Suntton p.101
# On-Policy First-Visit MC control(for e-soft policies) 최적의 정책 pi*를 찾는 방법
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

# 상태: (player 카드 합계, dealer 오픈 카드, 유용한 에이스 보유) 예) (6, 1, False)
win = 0
lose = 0
draw = 0
GAMMA = 1  # 할인 없음
# 알고리즘 매개변수: 작은 e > 0
e = 0.2
num_episodes = 100_000

env = gym.make("Blackjack-v1", sab=True, render_mode=None)
num_actions = env.action_space.n

# 초기화
# 임의의 e-soft 정책을 pi로 설정
# 모든 s, a에 대한 Q(s,a) 설정
# Returns(s, a)를 모든 s, a에 대해 빈 리스트로 초기화
pi = defaultdict(lambda: np.ones(num_actions, dtype=float) / num_actions)
Q = defaultdict(lambda: np.zeros(num_actions))
Returns = defaultdict(list)

# 무한히 반복 (각 에피소드에 대해)
for n_episode in range(num_episodes):
    # 정책 pi를 따라 에피소드 생성: S0,A0,R1,S1,A1,R2,..ST-1,AT-1,RT
    episode = []
    s, _ = env.reset()
    while True:
        P = pi[s]
        a = np.random.choice(np.arange(len(P)), p=P)  # 0:stick, 1:hit
        s_, r, terminated, truncated, _ = env.step(a)
        # s:(player가 가진 카드 합계, dealer 오픈 카드, 유용한 에이스 보유)
        episode.append((s, a, r))
        if terminated or truncated:
            if r == 1:
                win += 1
            elif r == -1:
                lose += 1
            else:
                draw += 1
            break
        s = s_

    # G <- 0
    G = 0
    # 에피소드의 각 단계에 대해 반복, t=T-1, T-2,...0
    visited_state_action_pair = []
    for s, a, r in episode[::-1]:
        # G <- gamma*G + R_(t+1)
        G = GAMMA * G + r
        # S_t, A_t 쌍이 S_0,A_0 S_1,A_1..S_(t-1),A_(t-1)에 나타나지 않는 경우:
        # G를 Returns(S_t, A_t)에 추가
        # Q(S_t,A_t) <- Returns(S_t, A_t)의 평균
        if (s, a) not in visited_state_action_pair:
            Returns[(s, a)].append(G)
            Q[s][a] = np.mean(Returns[(s, a)])
            visited_state_action_pair.append((s, a))

        # A* <- argmax_a Q(S_t,a)
        A_star = np.argmax(Q[s])
        # 모든 a에 대해:
        # a = A* 이면 pi(a|S_t) <- 1-e + e/|A(S_t)|
        # a != A* 이면 pi(a|S_t) <- e/|A(St)|
        for a in range(num_actions):
            if a == A_star:
                pi[s][a] = 1 - e + e/num_actions
            else:
                pi[s][a] = e/num_actions
                
    if n_episode % 1000 == 0:
        print(f"{n_episode}번째 에피소드 완료")

print("승리 비율 = {:.2f}%".format(win/num_episodes*100))
print("패배 비율 = {:.2f}%".format(lose/num_episodes*100))
print("무승부 비율 = {:.2f}%".format(draw/num_episodes*100))

#특정 상태(state)에서 가능한 모든 행동(actions)의 값 중에서 가장 큰 값(최적의 행동 가치)을 찾고, 
#그 값을 그 상태의 가치 함수 값으로 설정
V = defaultdict(float)
for state, actions in Q.items():
    # 해당 상태에서 가능한 모든 행동의 값 중 최대값을 찾음
    action_value = np.max(actions)
    # 그 값을 해당 상태의 가치 함수 값으로 설정
    V[state] = action_value

# 예측
sample_state = (21, 3, True)
# 해당 상태에서 최적의 행동을 찾음
optimal_action = np.argmax(Q[sample_state])
# 상태의 가치 값을 소수점 두 자리까지 출력하고, 최적의 행동이 'stick'인지 'hit'인지 출력
print("상태 {}의 가치 = {:.2f}".format(sample_state, V[sample_state]),
      "stick" if optimal_action == 0 else "hit")

sample_state = (4, 1, False)
optimal_action = np.argmax(Q[sample_state])
print("상태 {}의 가치 = {:.2f}".format(sample_state, V[sample_state]),
      "stick" if optimal_action == 0 else "hit")

sample_state = (14, 8, True)
optimal_action = np.argmax(Q[sample_state])
print("상태 {}의 가치 = {:.2f}".format(sample_state, V[sample_state]),
      "stick" if optimal_action == 0 else "hit")

X, Y = np.meshgrid(
    np.arange(1, 11),    # dealer가 open 한 카드
    np.arange(12, 22))   # player가 가진 카드 합계

# V[(player가 가진 카드 합계, dealer 오픈 카드, 유용한 에이스 보유)]
no_usable_ace = np.apply_along_axis(lambda idx: V[(idx[1], idx[0], False)],
                                    2, np.dstack([X, Y]))
usable_ace = np.apply_along_axis(lambda idx: V[(idx[1], idx[0], True)],
                                 2, np.dstack([X, Y]))

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3),
                               subplot_kw={'projection': '3d'})

ax0.plot_surface(X, Y, no_usable_ace, cmap=plt.cm.YlGnBu_r)
ax0.set_xlabel('Player Cards')
ax0.set_ylabel('Dealer open Cards')
ax0.set_zlabel('MC Estimated Value')
ax0.set_title('No Useable Ace')

ax1.plot_surface(X, Y, usable_ace, cmap=plt.cm.YlGnBu_r)
ax1.set_xlabel('Player Cards')
ax1.set_ylabel('Dealer open Cards')
ax1.set_zlabel('MC Estimated Value')
ax1.set_title('Useable Ace')

plt.show()
