import json
import numpy as np
from procon_env import ProconEnv
from agent import Agent
import torch
import os
import heapq
import time

def action_to_tuple(idx, N):
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
                mask[idx] = True
                idx += 1
    return mask

def count_pairs(field):
    N = field.shape[0]
    count = 0
    # Đếm cặp ngang
    for i in range(N):
        for j in range(N-1):
            if field[i, j] == field[i, j+1]:
                count += 1
    # Đếm cặp dọc
    for i in range(N-1):
        for j in range(N):
            if field[i, j] == field[i+1, j]:
                count += 1
    return count

def greedy_best_move(field, size):
    import numpy as np
    field = np.array(field, dtype=np.int32).reshape((size, size))
    base_pairs = count_pairs(field)
    best_pairs = base_pairs
    best_op = None
    for n in range(2, size+1):
        for y in range(size-n+1):
            for x in range(size-n+1):
                test_field = field.copy()
                sub = test_field[y:y+n, x:x+n].copy()
                test_field[y:y+n, x:x+n] = np.rot90(sub, -1)
                pairs = count_pairs(test_field)
                if pairs > best_pairs:
                    best_pairs = pairs
                    best_op = {"x": x, "y": y, "n": n}
    if best_op is not None:
        return [best_op]
    else:
        return []

def greedy_multi_steps(field, size):
    field = np.array(field, dtype=np.int32).reshape((size, size))
    ops = []
    while True:
        move = greedy_best_move(field, size)
        if not move:
            break
        op = move[0]
        ops.append(op)
        x, y, n = op['x'], op['y'], op['n']
        sub = field[y:y+n, x:x+n].copy()
        field[y:y+n, x:x+n] = np.rot90(sub, -1)
    return ops

def main():
    with open('input.json', 'r') as f:
        data = json.load(f)
    field_2d = data['problem']['field']['entities']
    size = data['problem']['field']['size']
    field = [item for row in field_2d for item in row]

    results = []
    total_start = time.time()
    
    choose = 2
    
    # --- Greedy ---
    if(choose == 1):
        start_time = time.time()
        ops_greedy = greedy_multi_steps(field, size)
        final_field = np.array(field, dtype=np.int32).reshape((size, size))
        for op in ops_greedy:
            x, y, n = op['x'], op['y'], op['n']
            sub = final_field[y:y+n, x:x+n].copy()
            final_field[y:y+n, x:x+n] = np.rot90(sub, -1)
        best_pair = count_pairs(final_field)
        num_steps = len(ops_greedy)
        elapsed = time.time() - start_time
        results.append((ops_greedy, best_pair, num_steps, elapsed, 'Greedy'))
        print(f"Greedy runtime: {elapsed:.3f} seconds | Best pair: {best_pair}, Steps: {num_steps}")

    # --- RL ---
    elif(choose == 2):
        env = ProconEnv(size=size, field=field)
        num_actions = get_num_actions(size)
        agent = Agent(size, num_actions, device='cpu')
        agent.load(f'dqn_model_{size}.pt')
        rl_results = []
        start_time = time.time()
        for run in range(100000):
            state, _ = env.reset()
            ops = []
            for _ in range(100):
                mask = valid_action_mask(env)
                action_idx = agent.select_action(state, valid_mask=mask)
                x, y, n = action_to_tuple(action_idx, size)
                ops.append({"x": int(x), "y": int(y), "n": int(n)})
                next_state, reward, done, truncated, _ = env.step((x, y, n))
                state = next_state
                if done:
                    break
            final_field = np.array(field, dtype=np.int32).reshape((size, size))
            for op in ops:
                x, y, n = op['x'], op['y'], op['n']
                sub = final_field[y:y+n, x:x+n].copy()
                final_field[y:y+n, x:x+n] = np.rot90(sub, -1)
            num_pairs = count_pairs(final_field)
            rl_results.append((ops, num_pairs, len(ops)))
        rl_results.sort(key=lambda x: (-x[1], x[2]))
        ops_rl = rl_results[0][0]
        best_pair = rl_results[0][1]
        num_steps = rl_results[0][2]
        elapsed = time.time() - start_time
        results.append((ops_rl, best_pair, num_steps, elapsed, 'RL'))
        print(f"RL runtime: {elapsed:.3f} seconds | Best pair: {best_pair}, Steps: {num_steps}")

    # --- Chọn kết quả tốt nhất ---
    results.sort(key=lambda x: (-x[1], x[2]))
    best_result = results[0]
    total_elapsed = time.time() - total_start
    print(f"\nBest overall: {best_result[4]} | Best pair: {best_result[1]}, Steps: {best_result[2]}")
    print(f"Total program runtime: {total_elapsed:.3f} seconds")

    with open('output.json', 'w') as f:
        json.dump({"ops": best_result[0]}, f)

    # In ra terminal
    print("Done! Số bước:", len(best_result[0]))
    print("Các phép xoay:", best_result[0])
    final_field = np.array(field, dtype=np.int32).reshape((size, size))
    for op in best_result[0]:
        x, y, n = op['x'], op['y'], op['n']
        sub = final_field[y:y+n, x:x+n].copy()
        final_field[y:y+n, x:x+n] = np.rot90(sub, -1)
    print("Số cặp cuối cùng:", count_pairs(final_field))
    print(final_field)

if __name__ == '__main__':
    main()
    
#   x: cột  ;  y: hàng  ;  n: kích thước