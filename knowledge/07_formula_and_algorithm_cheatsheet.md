# CHEATSHEET: TOÀN BỘ CÔNG THỨC TOÁN & BÍ KÍP THUẬT TOÁN MÔN REL

> **Mục đích**: Bảng tra cứu "bí kíp" siêu tốc tổng hợp mọi phương trình Bellman, quy tắc cập nhật tham số, công thức đạo hàm và bảng so sánh trực quan toàn bộ chương trình Reinforcement Learning.

---

## 1. Hệ thống Phương trình Bellman (The Bellman Equations)

| Loại Phương trình | Hàm Mục tiêu | Công thức Toán học Chuẩn tắc |
| :--- | :--- | :--- |
| **Bellman Expectation ($v_\pi$)** | Giá trị trạng thái theo chính sách | $$v_\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \big[ r + \gamma v_\pi(s') \big]$$ |
| **Bellman Expectation ($q_\pi$)** | Giá trị hành động theo chính sách | $$q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a) \Big[ r + \gamma \sum_{a'} \pi(a' \mid s') q_\pi(s', a') \Big]$$ |
| **Bellman Optimality ($v_*$)** | Giá trị trạng thái tối ưu | $$v_*(s) = \max_{a} \sum_{s', r} p(s', r \mid s, a) \big[ r + \gamma v_*(s') \big]$$ |
| **Bellman Optimality ($q_*$)** | Giá trị hành động tối ưu | $$q_*(s, a) = \sum_{s', r} p(s', r \mid s, a) \Big[ r + \gamma \max_{a'} q_*(s', a') \Big]$$ |
| **Differential Bellman ($v_\pi$)** | Continuing Task (Average Reward) | $$v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \big[ r - r(\pi) + v_\pi(s') \big]$$ |

---

## 2. Bảng So sánh & Quy tắc Cập nhật của các Thuật toán Core

| Thuật toán | Loại hình | TD Target ($U_t$) | Sai số TD ($\delta_t$) / Quy tắc Cập nhật |
| :--- | :--- | :--- | :--- |
| **TD(0)** | On-policy, Model-free | $R_{t+1} + \gamma V(S_{t+1})$ | $V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$ |
| **SARSA** | On-policy Control | $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ | $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$ |
| **Q-Learning** | Off-policy Control | $R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)$ | $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)]$ |
| **Expected SARSA** | On/Off-policy Control | $R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1}) Q(S_{t+1}, a)$ | $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[ R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \big]$ |
| **Linear Sarsa (Tile Coding)** | On-policy Control với FA | $R_{t+1} + \gamma \sum_{i \in \text{tiles}'} w[A_{t+1}, i]$ | $w[A_t, \text{tiles}] \leftarrow w[A_t, \text{tiles}] + \alpha \delta_t$ *(với $\alpha = \frac{\alpha_{\text{base}}}{m}$)* |
| **Actor-Critic (Continuing)** | Policy Gradient Control | $R_{t+1} - \bar{R}_t + \hat{v}(S_{t+1}, w)$ | $\delta_t = R_{t+1} - \bar{R}_t + \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)$ |

---

## 3. Bộ Ba Cập nhật Actor-Critic cho Continuing Tasks

$$\begin{aligned}
\text{Sai số TD Vi phân:} \quad & \delta_t = R_{t+1} - \bar{R}_t + \hat{v}(S_{t+1}, w_t) - \hat{v}(S_t, w_t) \\
\text{Cập nhật Phần thưởng TB:} \quad & \bar{R}_{t+1} = \bar{R}_t + \alpha_r \delta_t \\
\text{Cập nhật Trọng số Critic:} \quad & w_{t+1} = w_t + \alpha_w \delta_t \nabla \hat{v}(S_t, w_t) \\
\text{Cập nhật Tham số Actor:} \quad & \theta_{t+1} = \theta_t + \alpha_\theta \delta_t \nabla \ln \pi(A_t \mid S_t, \theta_t)
\end{aligned}$$

#### Với Chính sách Softmax Tuyến tính:
$$\pi(a \mid s) = \frac{\exp\left(\sum_{i \in \text{tiles}(s)} \theta[a, i]\right)}{\sum_b \exp\left(\sum_{i \in \text{tiles}(s)} \theta[b, i]\right)}$$
Đạo hàm đối với vector tham số của hành động $a$:
$$\nabla_{\theta[a, \text{tiles}]} \ln \pi(A_t \mid S_t) = \left( \mathbb{I}_{a = A_t} - \pi(a \mid S_t) \right)$$
Quy tắc cập nhật ma trận Actor $\theta$:
$$\theta[a, \text{tiles}] \leftarrow \theta[a, \text{tiles}] + \alpha_\theta \delta_t \left( \mathbb{I}_{a = A_t} - \pi(a \mid S_t) \right)$$

---

## 4. Công thức Bộ Tối ưu hóa Adam (Adaptive Moment Estimation)

$$\begin{aligned}
m_t &= \beta_m m_{t-1} + (1 - \beta_m) g_t \\
v_t &= \beta_v v_{t-1} + (1 - \beta_v) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - (\beta_m)^t} \\
\hat{v}_t &= \frac{v_t}{1 - (\beta_v)^t} \\
W_t &= W_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon_{\text{adam}}} \hat{m}_t
\end{aligned}$$

---

## 5. Bảng Ma trận Phân loại Bản chất các Thuật toán

| Thuật toán | Model-Free hay Model-Based? | Bootstrapping? | Sampling? | On-Policy hay Off-Policy? |
| :--- | :--- | :--- | :--- | :--- |
| **Dynamic Programming (DP)** | **Model-Based** | **CÓ** | **KHÔNG** | On-Policy (Eval) / Greedy (Value) |
| **Monte Carlo (MC)** | **Model-Free** | **KHÔNG** | **CÓ** | Cả On-Policy lẫn Off-Policy (với IS) |
| **TD(0)** | **Model-Free** | **CÓ** | **CÓ** | On-Policy |
| **SARSA** | **Model-Free** | **CÓ** | **CÓ** | **On-Policy** |
| **Q-Learning** | **Model-Free** | **CÓ** | **CÓ** | **Off-Policy** |
| **Expected SARSA** | **Model-Free** | **CÓ** | **CÓ** | Linh hoạt (Cả On lẫn Off-Policy) |
| **Dyna-Q / Dyna-Q+** | **Kết hợp (Model-Based & Free)** | **CÓ** | **CÓ** | Thường là Off-Policy (Q-learning planning) |
| **Actor-Critic** | **Model-Free** | **CÓ** | **CÓ** | **On-Policy** |

---

## 6. Các Hằng số & Quy tắc Ngón tay cái (Rules of Thumb)

1. **Tile Coding Step Size**:
   Nếu dùng $m$ tilings, luôn đặt $\alpha_{\text{tiling}} = \frac{\alpha}{m}$. Ví dụ: với $8$ tilings và mong muốn $\alpha = 0.5$, ta truyền `alpha = 0.5 / 8`.
2. **Kỹ thuật Log-Sum-Exp (Trừ Max)**:
   Khi tính Softmax, luôn lấy $c = \max_i z_i$ và tính $\frac{\exp(z_i - c)}{\sum_j \exp(z_j - c)}$ để ngăn chặn 100% nguy cơ tràn số `inf`/`nan`.
3. **Cờ Terminal State trong Deep RL**:
   Target luôn có dạng: $R + (1 - \text{terminals}) \cdot \gamma \cdot V(S')$. Nếu trạng thái tiếp theo là Terminal, tuyệt đối không được cộng dồn giá trị tương lai.
4. **Adam Products**:
   Không tính lũy thừa $(\beta)^t$ trực tiếp mà nhân dồn tích lũy `self.beta_m_product *= self.beta_m` tại mỗi bước cập nhật.
