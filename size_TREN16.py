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
            if j + 1 < n and v == row[j + 1]:
                pairs += 1
            if i + 1 < n and v == field[i + 1][j]:
                pairs += 1
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

def local_search_solver(start_field, time_limit=270, save_interval=10, max_steps=20000):
    start_time = time.time()
    last_save = start_time
    n = len(start_field)

    current_field = [r[:] for r in start_field]
    current_score = count_pairs(current_field)

    best_field = [r[:] for r in current_field]
    best_score = current_score
    best_ops = []
    current_ops = []

    sizes = ds_xoay
    if n >= 16: sizes.append(5)
    if n >= 24: sizes.append(6)

    moves = [(x,y,k)
             for k in sizes
             for y in range(n-k+1)
             for x in range(n-k+1)]

    if n <= 12: sample_moves = 35
    elif n <= 16: sample_moves = 50
    elif n <= 20: sample_moves = 60
    else: sample_moves = 80

    accept_bad_prob = 0.003

    tabu = []
    tabu_limit = 50

    no_improve = 0

    while True:
        now = time.time()
        elapsed = now - start_time

        if elapsed > time_limit:
            break

        if now - last_save >= save_interval:
            save_solution(best_ops, best_score)
            last_save = now

        if len(current_ops) >= max_steps:
            current_field = [r[:] for r in best_field]
            current_score = best_score
            current_ops = best_ops[:]
            tabu.clear()
            no_improve = 0
            continue

        candidates = []
        for _ in range(sample_moves):
            m = None
            tries = 0
            while True:
                m = random.choice(moves)
                tries += 1
                if m not in tabu or tries > 30:
                    break

            x,y,k = m
            nf = rotate_submatrix(current_field, x,y,k)
            ns = count_pairs(nf)
            candidates.append((ns,(x,y,k),nf))

        candidates.sort(key=lambda x: -x[0])

        best_ns, best_move, best_nf = candidates[0]
        rnd_ns, rnd_move, rnd_nf = random.choice(candidates)

        delta = best_ns - current_score

        accept = False
        if delta >= 0:
            accept = True
            used_move = best_move
            new_field = best_nf
            new_score = best_ns
        else:
            if random.random() < accept_bad_prob:
                accept = True
                used_move = rnd_move
                new_field = rnd_nf
                new_score = rnd_ns

        if accept:
            x,y,k = used_move
            current_field = new_field
            current_score = new_score
            current_ops.append({"x":x,"y":y,"n":k})

            tabu.append(used_move)
            if len(tabu) > tabu_limit:
                tabu.pop(0)

            no_improve += 1

            if new_score > best_score:
                best_score = new_score
                best_field = [r[:] for r in current_field]
                best_ops = current_ops[-max_steps:]
                no_improve = 0

            elif new_score == best_score:
                if len(current_ops) < len(best_ops):
                    best_field = [r[:] for r in current_field]
                    best_ops = current_ops[-max_steps:]

        # RESET nếu lâu không cải thiện
        if no_improve > 20000:
            current_field = [r[:] for r in best_field]
            current_score = best_score
            current_ops = best_ops[:]
            tabu.clear()
            no_improve = 0

    save_solution(best_ops, best_score)
    runtime = time.time() - start_time
    return best_ops, best_score, best_field, runtime

if __name__ == "__main__":
    with open("input.json","r",encoding="utf-8") as f:
        data = json.load(f)

    field = data["problem"]["field"]["entities"]

    ops, score, _, runtime = local_search_solver(
        field,
        time_limit=limit_time,
        save_interval=10,
        max_steps=20000,
    )
    
    print(f"Best score: {score} pairs, {len(ops)} steps")
