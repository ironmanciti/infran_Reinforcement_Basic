import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

stick_threshold = 17  # 스틱(threshold) 기준점 설정 (17 이상이면 더 이상 카드를 받지 않음)
win_cnt = 0  # 승리 횟수 초기화
lose_cnt = 0  # 패배 횟수 초기화
draw_cnt = 0  # 무승부 횟수 초기화
num_episodes = 100_000  # 에피소드 수 설정 (총 100,000번의 게임 진행)
GAMMA = 1  # 할인율 설정 (여기서는 할인 없음, 즉 미래 보상과 현재 보상을 동일하게 취급)

# Blackjack 환경 생성, ab=True를 설정하면 게임이 Sutton and Barto의 책에서 설명된 대로 정확하게 진행
env = gym.make("Blackjack-v1", sab=True)  

# 평가할 정책 pi
def pi(state):
    # state : (player 카드 합계, dealer 공개 카드, 사용 가능한 에이스 보유) ex) (6, 1, False)
    # player 카드가 stick_threshold 이상이면 무조건 stick 하고
    # 그렇지 않으면 hit 하는 전략
    #0:stick, 1:hit
    return 0 if state[0] >= stick_threshold else 1

# V(s) 초기화
V = defaultdict(float)   # 상태 가치 함수 V를 기본값 0으로 초기화
# 모든 s에 대해 Returns(s) <- 빈 리스트
Returns = defaultdict(list)  # 상태 s에 대한 반환 값을 저장할 빈 리스트 초기화

# 무한히 반복 (각 에피소드에 대해)
for i in range(num_episodes):
    # 정책 pi를 따르는 에피소드 생성: S0,A0,R1,S1,A1,R2,..ST-1,AT-1,RT
    episode = []
    s, _ = env.reset()
    while True:
        a = pi(s)  # 정책 pi 따름
        s_, r, terminated, truncated, _ = env.step(a)
        # s_ : (player의 hand 합계, dealer 공개 카드, 사용 가능한 에이스 보유)
        episode.append((s, a, r))
        if terminated or truncated:  # 에피소드 종료
            if r == 1:
                win_cnt += 1
            elif r == -1:
                lose_cnt += 1
            else:
                draw_cnt += 1
            break
        s = s_
    #G <- 0
    G = 0
    # 에피소드의 각 스텝에 대해 역순으로 반복: t=T-1,T-2,...,0
    visited_states = []
    for s, a, r in episode[::-1]:
        # G <- gamma*G + R_(t+1)
        G = GAMMA * G + r
        # S_0, S_1,...S_(t-1)에서 S_t가 등장하지 않는 경우:
        if s not in visited_states:
            # G를 Returns(S_t)에 추가
            Returns[s].append(G)
            # V(S_t) <- Returns(S_t)의 평균값
            V[s] = np.mean(Returns[s])
            # 방문한 상태로 추가
            visited_states.append(s)
    if i % 5000 == 0:
        print(f"episode {i} completed...")

print('stick threshold = {}'.format(stick_threshold))
print("win ratio = {:.2f}%".format(win_cnt/num_episodes*100))
print("lose ratio = {:.2f}%".format(lose_cnt/num_episodes*100))
print("draw ratio = {:.2f}%".format(draw_cnt/num_episodes*100))

sample_state = (21, 3, True)
print("state {}의 가치 = {:.2f}".format(sample_state, V[sample_state]))
print(
    f"     player가 손에 {sample_state[0]}를 들고 dealer가 {sample_state[1]}를 보여주고 있을 때")

sample_state = (14, 1, False)
print("state {}의 가치 = {:.2f}".format(sample_state, V[sample_state]))
print(
    f"     player가 손에 {sample_state[0]}를 들고 dealer가 {sample_state[1]}를 보여주고 있을 때")

#시각화
X, Y = np.meshgrid(
    np.arange(12, 22),   # player가 가진 카드 합계 (12~21)
    np.arange(1, 11))    # dealer가 공개한 카드 (1~10)

#V[(player의 hand 합계, dealer 공개 카드, 사용 가능한 에이스 보유)]
no_usable_ace = np.apply_along_axis(lambda idx: V[(idx[0], idx[1], False)],
                                    2, np.dstack([X, Y]))
usable_ace = np.apply_along_axis(lambda idx: V[(idx[0], idx[1], True)],
                                 2, np.dstack([X, Y]))

# 3D 그래프를 위한 플롯 생성
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4),
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
