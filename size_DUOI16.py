import json
import time
import random

limit_time = 270
ds_xoay = [2,3,4]


def count_pairs(field):
    n = len(field)
    pairs = 0
    for i in range(n):
        row = field[i]
        for j in range(n):
            v = row[j]
            if j + 1 < n and v == row[j + 1]: pairs += 1
            if i + 1 < n and v == field[i + 1][j]: pairs += 1
    return pairs

def rotate_submatrix(field, x, y, k):
    row0 = y
    col0 = x
    new_field = [r[:] for r in field]
    sub = [row[col0:col0+k] for row in field[row0:row0+k]]
    rotated = [list(r) for r in zip(*sub[::-1])]
    for i in range(k):
        for j in range(k):
            new_field[row0+i][col0+j] = rotated[i][j]
    return new_field

def save_solution(ops, score, filename="output.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"ops": ops}, f, ensure_ascii=False, indent=2)

def local_search_solver(
    start_field,
    time_limit=270,
    save_interval=5,
    max_steps=10000,
):
    start_time = time.time()
    last_save_time = start_time

    n = len(start_field)

    current_field = [r[:] for r in start_field]
    current_score = count_pairs(current_field)

    best_field = [r[:] for r in current_field]
    best_score = current_score
    best_ops = []
    current_ops = []

    allowed_sizes = ds_xoay
    
    moves = [(x,y,k)
             for k in allowed_sizes
             for y in range(n-k+1)
             for x in range(n-k+1)]

    if n <= 10: sample_moves = 20
    elif n <= 14: sample_moves = 30
    else: sample_moves = 40

    base_accept_prob = 0.02
    no_improve = 0
    iterations = 0

    while True:
        now = time.time()
        elapsed = now - start_time
        if elapsed > time_limit:
            break

        if now - last_save_time >= save_interval:
            save_solution(best_ops, best_score)
            last_save_time = now

        iterations += 1

        if len(current_ops) >= max_steps:
            current_field = [r[:] for r in best_field]
            current_score = best_score
            current_ops = best_ops[:]
            no_improve = 0
            continue

        candidate_moves = random.sample(moves, min(sample_moves, len(moves)))

        best_c_score = None
        best_c_move = None
        best_c_field = None

        random_move = None
        random_field = None
        random_score = None

        for (x,y,k) in candidate_moves:
            nf = rotate_submatrix(current_field, x,y,k)
            ns = count_pairs(nf)

            if random_field is None:
                random_move = (x,y,k)
                random_field = nf
                random_score = ns

            if (best_c_score is None) or (ns > best_c_score):
                best_c_score = ns
                best_c_move = (x,y,k)
                best_c_field = nf

        if best_c_field is None:
            continue

        delta = best_c_score - current_score
        t_ratio = max(0.0, 1 - elapsed / time_limit)
        accept_prob = base_accept_prob * (0.5 + 0.5 * t_ratio)

        accept = False
        if delta >= 0:
            accept = True
            used_move = best_c_move
            new_field = best_c_field
            new_score = best_c_score
        else:
            if random.random() < accept_prob:
                accept = True
                used_move = random_move
                new_field = random_field
                new_score = random_score

        if accept:
            x,y,k = used_move
            current_field = new_field
            current_score = new_score
            current_ops.append({"x":x,"y":y,"n":k})
            no_improve += 1

            if new_score > best_score:
                best_score = new_score
                best_field = [row[:] for row in current_field]
                best_ops = current_ops[-max_steps:]
                no_improve = 0

            elif new_score == best_score:
                if len(current_ops) < len(best_ops):
                    best_field = [row[:] for row in current_field]
                    best_ops = current_ops[-max_steps:]

        if no_improve > 5000:
            current_field = [r[:] for r in best_field]
            current_score = best_score
            current_ops = best_ops[:]
            no_improve = 0

    save_solution(best_ops, best_score)
    return best_ops, best_score, best_field, time.time() - start_time

if __name__ == "__main__":
    with open("input.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    field = data["problem"]["field"]["entities"]

    ops, score, final_field, runtime = local_search_solver(
        field,
        time_limit=limit_time,
        save_interval=10,
        max_steps=20000,
    )

    print(f"Best score: {score} pairs, {len(ops)} steps")
