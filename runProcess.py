import subprocess
import json
import os
import shutil

a = "size_DUOI16.py" ; b = "size_TREN16.py"

file_name = a
NUM_PROCESS = 2


def run_solver_instance(index):
    tmp_file = f"tmp_{index}.json"

    p = subprocess.Popen(["python", file_name, "--out", tmp_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p, tmp_file


def save_as_score_file(tmp_file):
    try:
        with open(tmp_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        score = data.get("score", 0)
        steps = len(data.get("ops", []))
        
        new_name = f"{score}_{steps}.json"
        suffix = 1

        while os.path.exists(new_name):
            new_name = f"{score}_{steps}_{suffix}.json"
            suffix += 1

        os.rename(tmp_file, new_name)
        print(f"✔ Saved → {new_name}")
        return score, new_name, steps

    except Exception as e:
        print(f"Failed to rename {tmp_file}: {e}")
        return -1, None, 99999999


def delete_all_except_final(final_file):
    for f in os.listdir("."):
        if f in ["input.json", final_file, "solver.py", "run_parallel.py"]:
            continue
        if f.endswith(".json") or f.startswith("tmp_"):
            try:
                os.remove(f)
            except:
                pass


def main():
    print(f"Running {NUM_PROCESS} solver processes in parallel...\n")

    processes = []
    for i in range(NUM_PROCESS):
        p, tf = run_solver_instance(i)
        processes.append((p, tf))

    scored_files = []

    for p, tf in processes:
        p.wait()
        score, fname, steps = save_as_score_file(tf)
        if fname:
            scored_files.append((score, fname, steps))

    if not scored_files:
        print("No results found.")
        return

    best_score, best_file, best_steps = max(
        scored_files,
        key=lambda x: (x[0], -x[2])
    )

    print(f"\nBEST: score = {best_score}, steps = {best_steps}")

    shutil.copy(best_file, "final.json")
    delete_all_except_final("final.json")


if __name__ == "__main__":
    main()
