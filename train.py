import numpy as np
import torch
from procon_env import ProconEnv
from agent import Agent
import os

# Tham số
SIZE = 8
EPISODES = 150000
TARGET_UPDATE = 20
MODEL_PATH = f'dqn_model_{SIZE}.pt'
CHECKPOINT_PATH = f'checkpoint_size{SIZE}.pth'

def action_to_tuple(idx, N):
    # idx -> (x, y, n) với n=2..N
    count = 0
    for n in range(2, N+1):
        for y in range(N-n+1):
            for x in range(N-n+1):
                if count == idx:
                    return (x, y, n)
                count += 1
    raise ValueError('Invalid action idx')

def tuple_to_action(x, y, n, N):
    count = 0
    for nn in range(2, N+1):
        for yy in range(N-nn+1):
            for xx in range(N-nn+1):
                if (x, y, n) == (xx, yy, nn):
                    return count
                count += 1
    raise ValueError('Invalid action tuple')

def get_num_actions(N):
    return sum((N-n+1)**2 for n in range(2, N+1))

def valid_action_mask(env):
    N = env.N
    mask = np.zeros(get_num_actions(N), dtype=bool)
    idx = 0
    for n in range(2, N+1):
        for y in range(N-n+1):
            for x in range(N-n+1):
                mask[idx] = True  # Tất cả action đều hợp lệ trong env này
                idx += 1
    return mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ProconEnv(size=SIZE)
    num_actions = get_num_actions(SIZE)
    agent = Agent(SIZE, num_actions, device=device)
    best_score = -float('inf')
    start_episode = 0
    # Load checkpoint nếu có
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        best_score = checkpoint.get('best_score', -float('inf'))
        start_episode = checkpoint['episode'] + 1
        print(f"Resuming from episode {start_episode}, best_score={best_score}, epsilon={agent.epsilon}")
    for episode in range(start_episode, EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        while not done:
            mask = valid_action_mask(env)
            action_idx = agent.select_action(state, valid_mask=mask)
            x, y, n = action_to_tuple(action_idx, SIZE)
            next_state, reward, done, truncated, _ = env.step((x, y, n))
            total_reward += reward
            agent.store((state, action_idx, reward, next_state, done))
            state = next_state
            steps += 1
            agent.optimize()
        if episode % TARGET_UPDATE == 0:
            agent.update_target()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        if total_reward > best_score:
            best_score = total_reward
            agent.save(MODEL_PATH)
        # Lưu checkpoint mỗi 100 episode (hoặc bạn có thể chỉnh lại tần suất)
        if episode % 100 == 0:
            checkpoint = {
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'target_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'best_score': best_score
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"Checkpoint saved at episode {episode}")
            print(f"Episode {episode}, Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.3f}, Best: {best_score}")
    print(f"Training done. Best model saved at {MODEL_PATH}")

if __name__ == '__main__':
    main() 