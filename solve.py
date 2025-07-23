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

def a_star_max_pairs(field, size, time_limit=60, max_steps=12):
    # time_limit tính bằng giây, max_steps là số bước tối đa
    import heapq, time
    start_time = time.time()
    field = np.array(field, dtype=np.int32).reshape((size, size))
    max_num = (size * size) // 2
    initial_pairs = count_pairs(field)
    heap = []
    visited = dict()  # field_bytes -> (num_pairs, num_steps)
    best = (initial_pairs, 0, [])
    # f = num_steps + heuristic (số cặp còn thiếu)
    heapq.heappush(heap, (0, -initial_pairs, 0, field.tobytes(), []))
    while heap:
        if time.time() - start_time > time_limit:
            print("A* timeout!")
            break
        f, neg_pairs, num_steps, field_bytes, path = heapq.heappop(heap)
        num_pairs = -neg_pairs
        if (num_pairs > best[0]) or (num_pairs == best[0] and num_steps < best[1]):
            best = (num_pairs, num_steps, path)
        if num_pairs == max_num:
            best = (num_pairs, num_steps, path)
            break
        if num_steps >= max_steps:
            continue
        cur_field = np.frombuffer(field_bytes, dtype=np.int32).reshape((size, size))
        for n in range(2, size+1):
            for y in range(size-n+1):
                for x in range(size-n+1):
                    next_field = cur_field.copy()
                    sub = next_field[y:y+n, x:x+n].copy()
                    next_field[y:y+n, x:x+n] = np.rot90(sub, -1)
                    next_bytes = next_field.tobytes()
                    next_pairs = count_pairs(next_field)
                    next_steps = num_steps + 1
                    # Chỉ push nếu trạng thái mới tốt hơn hoặc chưa từng thăm
                    if (next_bytes not in visited) or (next_pairs > visited[next_bytes][0]) or (next_pairs == visited[next_bytes][0] and next_steps < visited[next_bytes][1]):
                        visited[next_bytes] = (next_pairs, next_steps)
                        heuristic = max_num - next_pairs
                        f_score = next_steps + heuristic
                        heapq.heappush(heap, (f_score, -next_pairs, next_steps, next_bytes, path + [{"x": x, "y": y, "n": n}]))
        # Log tiến trình
        if len(visited) % 10000 == 0 and len(visited) > 0:
            print(f"Expanded {len(visited)} states, best pairs: {best[0]}, steps: {best[1]}")
    return best[2]

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
    # Đọc đúng định dạng đề bài
    field_2d = data['problem']['field']['entities']
    size = data['problem']['field']['size']
    # Chuyển field thành mảng 1 chiều
    field = [item for row in field_2d for item in row]
    if size == 4:
        ops = a_star_max_pairs(field, size)
    elif size == 6:
        ops = greedy_multi_steps(field, size)
    else:
        env = ProconEnv(size=size, field=field)
        num_actions = get_num_actions(size)
        agent = Agent(size, num_actions, device='cpu')
        agent.load(f'dqn_model_{size}.pt')
        agent.epsilon = 0  # Đảm bảo luôn chọn action tốt nhất khi inference
        state, _ = env.reset()
        ops = []
        for _ in range(100):
            mask = valid_action_mask(env)
            action_idx = agent.select_action(state, valid_mask=mask)
            x, y, n = action_to_tuple(action_idx, size)
            ops.append({"x": int(x), "y": int(y), "n": int(n)})
            next_state, reward, done, truncated, _ = env.step((x, y, n))
            state = next_state
            if done or reward < -50:
                break
    with open('output.json', 'w') as f:
        json.dump({"ops": ops}, f)
    print("Done! Số bước:", len(ops))
    print("Các phép xoay:", ops)
    final_field = np.array(field, dtype=np.int32).reshape((size, size))
    for op in ops:
        x, y, n = op['x'], op['y'], op['n']
        sub = final_field[y:y+n, x:x+n].copy()
        final_field[y:y+n, x:x+n] = np.rot90(sub, -1)
    print("Số cặp cuối cùng:", count_pairs(final_field))
    print(final_field)

if __name__ == '__main__':
    main()
    
#   x: cột  ;  y: hàng  ;  n: kích thước