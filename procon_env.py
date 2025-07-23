import gymnasium as gym
import numpy as np

class ProconEnv(gym.Env):
    """
    Môi trường RL cho bài toán Procon 2025.
    Observation: ma trận NxN các số nguyên.
    Action: (x, y, n) - xoay vùng vuông n x n tại (x, y) 90 độ CW.
    Reward: số cặp liền kề tăng lên sau mỗi bước.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, size=4, field=None):
        super().__init__()
        self.size = size
        self.N = size
        self.max_num = (self.N * self.N) // 2
        if field is not None:
            self.field = np.array(field, dtype=np.int32).reshape((self.N, self.N))
        else:
            self.field = self._generate_field()
        self.action_space = gym.spaces.MultiDiscrete([self.N, self.N, self.N-1])
        self.observation_space = gym.spaces.Box(low=0, high=self.max_num-1, shape=(self.N, self.N), dtype=np.int32)
        self._update_pairs()
        self.steps = 0
        self.max_steps = 200

    def _generate_field(self):
        nums = np.repeat(np.arange(self.max_num), 2)
        np.random.shuffle(nums)
        return nums.reshape((self.N, self.N))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if hasattr(self, 'init_field') and self.init_field is not None:
            self.field = self.init_field.copy()
        else:
            self.field = self._generate_field()
        self._update_pairs()
        self.steps = 0
        return self.field.copy(), {}

    def step(self, action):
        x, y, n = action
        n = n + 2  # n=0 -> 2x2, n=1 -> 3x3, ...
        if x < 0 or y < 0 or x + n > self.N or y + n > self.N:
            # Hành động không hợp lệ
            return self.field.copy(), -100, True, False, {}
        prev_pairs = self._count_pairs()
        # Xoay vùng
        sub = self.field[y:y+n, x:x+n]
        self.field[y:y+n, x:x+n] = np.rot90(sub, -1)
        self._update_pairs()
        new_pairs = self._count_pairs()
        reward = new_pairs - prev_pairs
        self.steps += 1
        done = (new_pairs == self.max_num) or (self.steps >= self.max_steps)
        truncated = False
        return self.field.copy(), reward, done, truncated, {}

    def _update_pairs(self):
        self.paired = np.zeros_like(self.field, dtype=bool)
        N = self.N
        for i in range(N):
            for j in range(N):
                v = self.field[i, j]
                for dx, dy in [(0,1),(1,0)]:
                    ni, nj = i+dy, j+dx
                    if ni<N and nj<N and self.field[ni, nj]==v:
                        self.paired[i, j] = True
                        self.paired[ni, nj] = True

    def _count_pairs(self):
        N = self.N
        count = 0
        for i in range(N):
            for j in range(N):
                v = self.field[i, j]
                for dx, dy in [(0,1),(1,0)]:
                    ni, nj = i+dy, j+dx
                    if ni<N and nj<N and self.field[ni, nj]==v:
                        count += 1
        return count // 2  # mỗi cặp đếm 2 lần

    def render(self):
        print(self.field) 