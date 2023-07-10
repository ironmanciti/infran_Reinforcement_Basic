# Environment 초기화
import gymnasium as gym
env = gym.make('CartPole-v1', render_mode="human")
obs, info = env.reset(seed=42)

# 시각화
for _ in range(10000):
    # take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
        
env.close()
