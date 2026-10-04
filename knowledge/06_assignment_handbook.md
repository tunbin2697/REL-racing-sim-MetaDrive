# CẨM NANG THỰC CHIẾN & GIẢI MÃ 3 BÀI ASSIGNMENT MÔN REL

> **Tài nguyên sinh viên**: Thư mục `Assignment-student/`  
> **Nội dung cẩm nang**: Hướng dẫn chi tiết từng bước, cấu trúc thuật toán, công thức toán học đối ứng, giải thích các khối code cần điền (`### START CODE HERE`), các bẫy thường gặp (pitfalls) và tiêu chí kiểm thử vượt qua 100% Graded Test Cases.

---

## 1. Assignment 1: Deep Expected Sarsa Agent (Lunar Lander)

- **Mục tiêu**: Lập trình hoàn chỉnh một tác nhân Học tăng cường sâu (Deep RL Agent) sử dụng mạng nơ-ron xấp xỉ giá trị hành động, bộ nhớ đệm kinh nghiệm Replay Buffer, bộ tối ưu hóa thích nghi Adam, và chính sách Softmax để điều khiển tàu vũ trụ hạ cánh thành công trong game **LunarLander-v2** (OpenAI Gym).
- **File notebook**: `Assignment 1. Implement your agent-Student.ipynb`

```
                                    +-----------------------+
                                    | LunarLander-v2 Env    |
                                    +-----------------------+
                                       ^                 |
                                Action |                 | (S, R, terminal, S')
                                       |                 v
+----------------------------------------------------------------------------------+
| AGENT                                                                            |
|                                                                                  |
|   +-------------------+  Sample Minibatch   +--------------------------------+   |
|   |  Replay Buffer    | ----------------->  |   Target Calculation           |   |
|   | (S, A, R, term, S')                     | y = R + (1-term)*gamma*E[Q_fixed]  |   |
|   +-------------------+                     +--------------------------------+   |
|                                                            |                     |
|                                                            v TD Error            |
|   +-------------------+   Gradient Update   +--------------------------------+   |
|   |    Adam Optimizer | <------------------ |   Semi-Gradient Backprop       |   |
|   +-------------------+                     |   delta = y - Q(S, A; network) |   |
|             |                               +--------------------------------+   |
|             v Cập nhật Weights                             ^                     |
|   +----------------------------------+                     |                     |
|   | ActionValueNetwork Q(s, a; W)    | --------------------+                     |
|   +----------------------------------+                                           |
|             | Q-values                                                           |
|             v                                                                    |
|   +----------------------------------+                                           |
|   | Softmax Action Selection (tau)   | ---> Action A_t                           |
|   +----------------------------------+                                           |
+----------------------------------------------------------------------------------+
```

---

### 1.1. Chi tiết các phần Code cần Hoàn thiện

#### Cell 10: `Adam.__init__` và `Adam.update_weights`
- **Mục đích**: Tự lập trình thuật toán tối ưu Adam để cập nhật trọng số mạng nơ-ron thay vì dùng PyTorch/TensorFlow.
- **Phần 1: Khởi tạo trong `__init__`**:
  Khởi tạo các moment $m$ và $v$ của tất cả các tầng về ma trận 0 với kích thước tương ứng:
  ```python
  for i in range(len(layer_sizes) - 1):
      self.m[i]["W"] = np.zeros((layer_sizes[i], layer_sizes[i + 1]))
      self.m[i]["b"] = np.zeros((1, layer_sizes[i + 1]))
      self.v[i]["W"] = np.zeros((layer_sizes[i], layer_sizes[i + 1]))
      self.v[i]["b"] = np.zeros((1, layer_sizes[i + 1]))
  ```
- **Phần 2: Cập nhật trong `update_weights`**:
  Với mỗi tầng $i$ và mỗi tham số $p \in [``W'', ``b'']$:
  1. Cập nhật tích số lũy thừa:
     ```python
     self.beta_m_product *= self.beta_m
     self.beta_v_product *= self.beta_v
     ```
  2. Cập nhật moment bậc 1 và bậc 2:
     ```python
     self.m[i][param] = self.beta_m * self.m[i][param] + (1.0 - self.beta_m) * g
     self.v[i][param] = self.beta_v * self.v[i][param] + (1.0 - self.beta_v) * (g ** 2)
     ```
  3. Hiệu chỉnh độ chệch (Bias correction):
     ```python
     m_hat = self.m[i][param] / (1.0 - self.beta_m_product)
     v_hat = self.v[i][param] / (1.0 - self.beta_v_product)
     ```
  4. Cập nhật trọng số:
     ```python
     weights[i][param] += self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon_adam)
     ```
     *(Chú ý: `self.step_size` trong code đã bao gồm dấu trừ hoặc công thức cập nhật tương ứng theo gradient)*.

---

#### Cell 18: Hàm `softmax(action_values, tau=1.0)`
- **Mục đích**: Chuyển đổi ma trận Q-values có kích thước `(batch_size, num_actions)` thành ma trận phân phối xác suất mà không bao giờ bị lỗi tràn số `NaN` / `Inf`.
- **Code thực hiện**:
  ```python
  # 1. Tính preferences chia cho nhiệt độ tau
  preferences = action_values / tau
  
  # 2. Tìm giá trị preference lớn nhất trên mỗi hàng (trục hành động)
  max_preference = np.max(preferences, axis=-1)
  
  # 3. Reshape để NumPy broadcast trừ max trên mỗi hàng
  reshaped_max_preference = max_preference.reshape((-1, 1))
  
  # 4. Tính tử số số mũ sau khi trừ max
  exp_preferences = np.exp(preferences - reshaped_max_preference)
  
  # 5. Tính mẫu số bằng tổng theo hàng
  sum_of_exp_preferences = np.sum(exp_preferences, axis=-1)
  
  # 6. Reshape mẫu số và tính phân phối xác suất
  reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
  action_probs = exp_preferences / reshaped_sum_of_exp_preferences
  ```

---

#### Cell 24: Hàm `get_td_error(...)`
- **Mục đích**: Tính vector sai số TD $\delta$ cho minibatch theo công thức **Expected Sarsa**:
  $$\delta_i = R_i + \gamma (1 - \text{terminal}_i) \sum_a \pi(a \mid S'_i) Q_{\text{target}}(S'_i, a) - Q(S_i, A_i)$$
- **Code thực hiện**:
  ```python
  # 1. Dự đoán Q tại trạng thái tiếp theo từ mạng cố định current_q
  q_next_mat = current_q.get_action_values(next_states)
  
  # 2. Tính xác suất chọn hành động tại trạng thái tiếp theo
  probs_mat = softmax(q_next_mat, tau)
  
  # 3. Tính kỳ vọng giá trị trạng thái kế tiếp: sum(pi * Q)
  v_next_vec = np.sum(probs_mat * q_next_mat, axis=-1)
  
  # 4. Triệt tiêu giá trị tương lai nếu trạng thái kế tiếp là Terminal
  v_next_vec = (1.0 - terminals) * v_next_vec
  
  # 5. Dự đoán Q tại trạng thái hiện tại từ network đang học
  q_mat = network.get_action_values(states)
  
  # 6. Lấy giá trị Q tương ứng với hành động đã thực hiện
  batch_indices = np.arange(states.shape[0])
  q_vec = q_mat[batch_indices, actions]
  
  # 7. Tính sai số TD delta
  delta_vec = rewards + discount * v_next_vec - q_vec
  ```

---

#### Cell 29: Hàm `optimize_network(...)`
- **Mục đích**: Tính gradient lan truyền ngược từ ma trận sai số TD và cập nhật trọng số mạng qua Adam.
- **Code thực hiện**:
  ```python
  # Lan truyền ngược sai số qua mạng nơ-ron
  td_update = network.get_TD_update(states, delta_mat)
  
  # Cập nhật trọng số của network bằng Adam optimizer
  optimizer.update_weights(network.weights, td_update)
  ```

---

#### Cell 33: Lớp `Agent` (`agent_step` và `agent_end`)
- **Trong `agent_step(self, reward, state)`**:
  1. Thêm mẫu mới vào Replay Buffer:
     ```python
     self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)
     ```
  2. Thực hiện cập nhật Replay:
     ```python
     for _ in range(self.num_replay):
         experiences = self.replay_buffer.sample()
         optimize_network(experiences, self.discount, self.optimizer, self.network, self.current_q, self.tau)
     ```
  3. Chọn hành động kế tiếp bằng `softmax`:
     ```python
     action_values = self.network.get_action_values(state)
     probs = softmax(action_values, self.tau).squeeze()
     action = self.rand_generator.choice(self.num_actions, p=probs)
     ```
  4. Lưu trạng thái và trả về `action`.
- **Trong `agent_end(self, reward)`**:
  1. Thêm mẫu kết thúc vào Replay Buffer (chú ý cờ `terminal = 1`):
     ```python
     self.replay_buffer.append(self.last_state, self.last_action, reward, 1, np.zeros_like(self.last_state))
     ```
  2. Thực hiện vòng lặp Replay tương tự như trong `agent_step`.

---

## 2. Assignment 2: Function Approximation and Control (Mountain Car)

- **Mục tiêu**: Giải bài toán xe leo dốc Mountain Car bằng thuật toán Linear Episodic Semi-gradient Sarsa kết hợp Tile Coding.
- **File notebook**: `Assignment 2- Function Approximation and Control-Student.ipynb`

### 2.1. Chi tiết các phần Code cần Hoàn thiện

#### Cell 9: Lớp `MountainCarTileCoder.get_tiles(self, position, velocity)`
- **Khoảng giá trị của Mountain Car**:
  - Vị trí (Position): $[-1.2, 0.5]$
  - Vận tốc (Velocity): $[-0.07, 0.07]$
- **Code thực hiện**:
  ```python
  # 1. Tính hệ số tỷ lệ hóa (scale factor) về miền [0, num_tiles]
  position_scale = self.num_tiles / (0.5 - (-1.2))
  velocity_scale = self.num_tiles / (0.07 - (-0.07))
  
  # 2. Chuẩn hóa vị trí và vận tốc
  scaled_position = (position - (-1.2)) * position_scale
  scaled_velocity = (velocity - (-0.07)) * velocity_scale
  
  # 3. Lấy danh sách chỉ số tiles kích hoạt từ thư viện tc.tiles
  tiles = tc.tiles(self.iht, self.num_tilings, [scaled_position, scaled_velocity])
  return np.array(tiles)
  ```

---

#### Cell 13: Lớp `SarsaAgent`
- **Hàm `select_action(self, tiles)`**:
  Tính giá trị $Q(s, a) = \sum_{i \in \text{tiles}} w[a, i]$ cho cả 3 hành động $a \in [0, 1, 2]$:
  ```python
  action_values = []
  for a in range(self.num_actions):
      action_values.append(np.sum(self.w[a, tiles]))
  
  # Chọn hành động theo epsilon-greedy (có xử lý tie-breaking ngẫu nhiên)
  if self.rand_generator.rand() < self.epsilon:
      action = self.rand_generator.choice(self.num_actions)
  else:
      action = self.argmax(action_values)
  return action, action_values[action]
  ```

- **Hàm `agent_step(self, reward, state)`**:
  ```python
  # Lấy active tiles và chọn hành động kế tiếp
  tiles = self.mctc.get_tiles(state[0], state[1])
  action, current_q = self.select_action(tiles)
  
  # Tính TD error delta: R + gamma * Q(S', A') - Q(S, A)
  last_q = np.sum(self.w[self.last_action, self.previous_tiles])
  delta = reward + self.gamma * current_q - last_q
  
  # Cập nhật trọng số cho các tiles của hành động trước
  self.w[self.last_action, self.previous_tiles] += self.alpha * delta
  
  # Cập nhật biến trạng thái
  self.previous_tiles = tiles
  self.last_action = action
  return action
  ```

- **Hàm `agent_end(self, reward)`**:
  Tại trạng thái kết thúc, $Q(S', A') = 0$:
  ```python
  last_q = np.sum(self.w[self.last_action, self.previous_tiles])
  delta = reward - last_q
  self.w[self.last_action, self.previous_tiles] += self.alpha * delta
  ```

---

## 3. Assignment 3: Continuing Average Reward Softmax Actor-Critic (Pendulum)

- **Mục tiêu**: Lập trình thuật toán Actor-Critic trên bài toán liên tục không hồi kết (Continuing Task) với công thức phần thưởng trung bình (Average Reward) và chính sách Softmax điều khiển con lắc ngược Pendulum Swing-Up.
- **File notebook**: `Assignment 3-Average Reward Softmax Actor-Critic-Student.ipynb`

### 3.1. Chi tiết các phần Code cần Hoàn thiện

#### Cell 5: Lớp `PendulumTileCoder.get_tiles(self, angle, ang_vel)`
- **Khoảng giá trị của Pendulum**:
  - Góc quay (Angle): $[-\pi, \pi]$
  - Vận tốc góc (Angular Velocity): $[-2\pi, 2\pi]$
- **Code thực hiện**:
  ```python
  angle_scale = self.num_tiles / (np.pi - (-np.pi))
  ang_vel_scale = self.num_tiles / (2.0 * np.pi - (-2.0 * np.pi))
  
  scaled_angle = (angle - (-np.pi)) * angle_scale
  scaled_ang_vel = (ang_vel - (-2.0 * np.pi)) * ang_vel_scale
  
  tiles = tc.tiles(self.iht, self.num_tilings, [scaled_angle, scaled_ang_vel])
  return np.array(tiles)
  ```

---

#### Cell 11: Hàm `compute_softmax_prob(actor_w, tiles)`
- **Mục đích**: Tính phân phối xác suất $\pi(a \mid s)$ từ ma trận trọng số Actor kích thước `(num_actions, iht_size)`.
- **Code thực hiện**:
  ```python
  # 1. Tính độ ưu tiên hành động h(s, a) = sum(actor_w[a, tiles])
  state_action_preferences = []
  for a in range(actor_w.shape[0]):
      state_action_preferences.append(np.sum(actor_w[a, tiles]))
  state_action_preferences = np.array(state_action_preferences)
  
  # 2. Tìm giá trị lớn nhất c để trừ max
  c = np.max(state_action_preferences)
  
  # 3. Tính số mũ và chuẩn hóa thành xác suất
  exp_preferences = np.exp(state_action_preferences - c)
  softmax_prob = exp_preferences / np.sum(exp_preferences)
  return softmax_prob
  ```

---

#### Cell 15: Lớp `ActorCriticSoftmaxAgent.agent_step(self, reward, state)`
- **Bộ 3 cập nhật đồng thời trong Continuing Tasks**:
  ```python
  # 1. Lấy tiles của trạng thái hiện tại và phân phối xác suất
  tiles = self.tc.get_tiles(state[0], state[1])
  probs = compute_softmax_prob(self.actor_w, tiles)
  
  # 2. Chọn hành động A' ngẫu nhiên theo phân phối xác suất probs
  action = self.rand_generator.choice(self.actions, p=probs)
  
  # 3. Tính giá trị trạng thái từ Critic: v(S) và v(S')
  v_current = np.sum(self.critic_w[self.prev_tiles])
  v_next = np.sum(self.critic_w[tiles])
  
  # 4. Tính Sai số TD Vi phân (Differential TD Error)
  # Chú ý: Continuing Task có gamma = 1 và trừ đi avg_reward!
  delta = reward - self.avg_reward + v_next - v_current
  
  # 5. Cập nhật Ước lượng Phần thưởng Trung bình
  self.avg_reward += self.avg_reward_step_size * delta
  
  # 6. Cập nhật Trọng số Critic
  self.critic_w[self.prev_tiles] += self.critic_step_size * delta
  
  # 7. Cập nhật Trọng số Actor cho toàn bộ các hành động
  for a in self.actions:
      # Nếu a là hành động vừa thực hiện thì chỉ số I = 1, ngược lại = 0
      indicator = 1.0 if a == self.last_action else 0.0
      grad_ln_pi = indicator - self.prev_probs[a]
      self.actor_w[a, self.prev_tiles] += self.actor_step_size * delta * grad_ln_pi
      
  # 8. Cập nhật biến lưu trữ cho bước kế tiếp
  self.prev_tiles = tiles
  self.last_action = action
  self.prev_probs = probs
  
  return action
  ```

---

## 4. Bảng Tổng hợp Pitfalls & Lưu ý Sống còn khi Chấm điểm

| Bài tập | Vị trí | Lỗi thường gặp | Cách khắc phục chuẩn xác |
| :--- | :--- | :--- | :--- |
| **Assignment 1** | `get_td_error` | Quên nhân `(1 - terminals)` với `v_next_vec` | Khi tàu chạm đất/nổ tung, trạng thái tiếp theo không còn tương lai $\implies V(S') = 0$! |
| **Assignment 1** | `softmax` | Không trừ `max_preference` dẫn đến tràn số `inf` | Luôn dùng mẹo Log-Sum-Exp: `np.exp(x - max_x)`. |
| **Assignment 1** | `Adam` | Nhầm lũy thừa bước thời gian $t$ | Timestep chạy từ $1$, mẫu số là `1.0 - product`, không được để mẫu số bằng 0. |
| **Assignment 2** | `MountainCarTileCoder` | Nhân chia sai tỷ lệ scale miền giá trị | Công thức chuẩn: `(val - min_val) / (max_val - min_val) * num_tiles`. |
| **Assignment 2** | `SarsaAgent` | Quên cập nhật episode kết thúc trong `agent_end` | Ở `agent_end`, target chỉ còn là `R` (không còn term $\gamma Q$). |
| **Assignment 3** | `agent_step` | Dùng $\gamma$ chiết khấu hoặc quên trừ $\bar{R}$ | Continuing Tasks **không dùng $\gamma$** ($\gamma = 1$), công thức TD error bắt buộc là $R - \bar{R} + V' - V$. |
| **Assignment 3** | `compute_softmax` | Truyền nhầm thứ tự trục của ma trận `actor_w` | `actor_w` có shape `(num_actions, iht_size)`. Chỉ số hành động $a$ nằm ở hàng đầu tiên! |
