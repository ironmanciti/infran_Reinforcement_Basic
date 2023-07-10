import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict



# 평가할 정책 pi의 입력


def pi(state):
    # state : (player 카드 합계, dealer 공개 카드, 사용 가능한 에이스 보유) ex) (6, 1, False)
    # player 카드가 stick_threshold 이상이면 무조건 stick 하고
    # 그렇지 않으면 hit 하는 전략
    #0:stick, 1:hit



# V(s) 초기화
# 모든 s에 대해 Returns(s) <- 빈 리스트


# 무한히 반복 (각 에피소드에 대해)
for i in range(num_episodes):
    # 정책 pi를 따르는 에피소드 생성: S0,A0,R1,S1,A1,R2,..ST-1,AT-1,RT
   
    while True:
      
        # s_ : (player의 hand 합계, dealer 공개 카드, 사용 가능한 에이스 보유)
        
    #G <- 0
    
    #에피소드의 각 스텝에 대해 반복: t=T-1,T-2,...,0
    for s, a, r in episode[::-1]:
        #G <- gamma*G + R_(t+1)
        
        # S_0, S_1,...S_(t-1)에서 S_t가 등장하지 않는 경우:
        if s not in visited_states:
            # G를 Returns(S_t)에 추가
            
            #V(S_t) <- average(Returns(S_t))
            


print('stick threshold = {}'.format(stick_threshold))


sample_state = (21, 3, True)
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


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4),
                               subplot_kw={'projection': '3d'})

ax0.plot_surface(X, Y, no_usable_ace, cmap=plt.cm.YlGnBu_r)
ax0.set_xlabel('Dealer open Cards')
ax0.set_ylabel('Player Cards')
ax0.set_zlabel('MC Estimated Value')
ax0.set_title('No Useable Ace')

plt.show()
