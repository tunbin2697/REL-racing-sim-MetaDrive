# MODULE 1: NỀN TẢNG TABULAR RL, MARKOV DECISION PROCESS & QUY HOẠCH ĐỘNG (DP)

> **Tài liệu tương ứng từ bài giảng**: Slide `1.1` đến `1.12`.  
> **Mục tiêu**: Cung cấp cơ sở toán học vững chắc nhất của Học tăng cường (Reinforcement Learning - RL), từ bài toán Bandit đơn giản, không gian quá trình quyết định Markov (MDP), các phương trình Bellman nền tảng, đến các giải thuật tối ưu bằng Quy hoạch động (Dynamic Programming).

---

## 1. Bài toán K-Armed Bandit (Slide 1.1 – 1.3)

### 1.1. Bản chất và Định nghĩa
Bài toán K-Armed Bandit là mô hình đơn giản nhất của RL:
- Có $k$ hành động khác nhau (tượng trưng cho $k$ cần gạt của máy đánh bạc).
- Mỗi khi người chơi chọn một hành động $A_t = a$, môi trường trả về một phần thưởng số (reward) $R_t$ từ một phân phối xác suất cố định nhưng chưa biết:
  $$q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]$$
  với $q_*(a)$ là **giá trị thực sự** (true value) của hành động $a$.

### 1.2. Ước lượng Giá trị Hành động (Action-Value Estimation)
Vì chưa biết $q_*(a)$, tác nhân cần ước lượng giá trị $Q_t(a) \approx q_*(a)$.

#### Phương pháp Trung bình Mẫu (Sample-Average):
$$Q_t(a) \doteq \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{I}_{A_i = a}}{\sum_{i=1}^{t-1} \mathbb{I}_{A_i = a}}$$

#### Công thức Cập nhật Tăng dần (Incremental Update Rule):
Thay vì lưu trữ toàn bộ lịch sử phần thưởng, ta cập nhật $Q$ ngay lập tức khi nhận phần thưởng thứ $n$:
$$Q_{n+1} = Q_n + \frac{1}{n} \left[ R_n - Q_n \right]$$
Dạng tổng quát của mọi quy tắc học trong RL:
$$\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} \times \left[ \text{Target} - \text{OldEstimate} \right]$$
Trong đó:
- $\left[ \text{Target} - \text{OldEstimate} \right]$ là **sai số dự đoán** (prediction error).
- Hệ số học $\alpha$: Với bài toán tĩnh (stationary), $\alpha = \frac{1}{n}$. Với bài toán biến thiên theo thời gian (non-stationary), ta dùng hằng số $\alpha \in (0, 1]$ (exponential recency-weighted average):
  $$Q_{n+1} = Q_n + \alpha [R_n - Q_n] = (1-\alpha)^n Q_1 + \sum_{i=1}^n \alpha(1-\alpha)^{n-i} R_i$$

### 1.3. Đánh đổi Khám phá và Khai thác (Exploration vs. Exploitation)
Tại mỗi bước thời gian:
- **Khai thác (Exploitation)**: Chọn hành động tham lam có giá trị ước lượng cao nhất $A_t^* = \arg\max_a Q_t(a)$ để tối đa hóa phần thưởng tức thời.
- **Khám phá (Exploration)**: Chọn các hành động phi tham lam để thu thập thêm thông tin, cải thiện độ chính xác của các ước lượng khác.

#### Các phương pháp cân bằng:
1. **$\epsilon$-greedy Selection**:
   $$\pi(a) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|}, & \text{nếu } a = \arg\max_{a'} Q(a') \\ \frac{\epsilon}{|\mathcal{A}|}, & \text{nếu } a \neq \arg\max_{a'} Q(a') \end{cases}$$
2. **Khởi tạo giá trị lạc quan (Optimistic Initial Values)**:
   - Đặt $Q_1(a) = +5$ (rất cao so với reward thực tế).
   - Tác nhân sẽ bị thất vọng sau mỗi lần thử, từ đó tự động khám phá toàn bộ các nhánh hành động khác trong giai đoạn đầu.
3. **Upper-Confidence-Bound (UCB)**:
   $$A_t \doteq \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$
   với $N_t(a)$ là số lần hành động $a$ đã được chọn. Càng ít được chọn, số hạng căn bậc hai càng lớn, thúc đẩy tác nhân khám phá nhánh còn mập mờ.

---

## 2. Quá trình Quyết định Markov (MDP) (Slide 1.4 – 1.6)

### 2.1. Vòng lặp Tương tác Tác nhân - Môi trường (Agent-Environment Loop)
Tại mỗi bước thời gian rời rạc $t = 0, 1, 2, \dots$:
1. Tác nhân quan sát trạng thái hiện tại $S_t \in \mathcal{S}$.
2. Dựa trên trạng thái đó, tác nhân chọn hành động $A_t \in \mathcal{A}(S_t)$.
3. Môi trường chuyển sang trạng thái mới $S_{t+1} \in \mathcal{S}$ và trả về phần thưởng $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$.

```
      +------------- Action A_t ------------->
      |                                      |
   [AGENT]                               [ENVIRONMENT]
      ^                                      |
      |-- State S_{t+1}, Reward R_{t+1} -----+
```

### 2.2. Tính chất Markov & Hàm Động lực học (Dynamics Function)
Một trạng thái $S_t$ có **tính chất Markov** nếu nó chứa đựng đầy đủ toàn bộ thông tin lịch sử cần thiết để xác định tương lai:
$$\mathbb{P}(S_{t+1}=s', R_{t+1}=r \mid S_t, A_t, R_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0) = \mathbb{P}(S_{t+1}=s', R_{t+1}=r \mid S_t, A_t)$$

Hàm động lực học 4 biến của MDP:
$$p(s', r \mid s, a) \doteq \mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a)$$
Từ hàm $p(s', r \mid s, a)$, ta suy ra:
- Xác suất chuyển trạng thái: $p(s' \mid s, a) = \sum_{r \in \mathcal{R}} p(s', r \mid s, a)$
- Phần thưởng kỳ vọng tức thời: $r(s, a) \doteq \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a] = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r \mid s, a)$

### 2.3. Lợi nhuận (Return) & Bài toán Có hồi kết vs. Liên tục
- **Mục tiêu của RL**: Tối đa hóa tổng phần thưởng kỳ vọng trong tương lai, gọi là **Lợi nhuận** (Return) $G_t$.
- **Episodic Tasks (Bài toán theo hồi/tập)**: Có trạng thái kết thúc (Terminal State $T$).
  $$G_t \doteq R_{t+1} + R_{t+2} + \dots + R_T = \sum_{k=0}^{T-t-1} R_{t+k+1}$$
- **Continuing Tasks (Bài toán liên tục)**: Diễn ra vô hạn ($T = \infty$). Để tránh $G_t \to \infty$, ta đưa vào **Hệ số chiết khấu (Discount Factor)** $\gamma \in [0, 1)$:
  $$G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
- **Quan hệ Đệ quy then chốt**:
  $$G_t = R_{t+1} + \gamma G_{t+1}$$

---

## 3. Chính sách, Hàm Giá trị & Phương trình Bellman (Slide 1.7 – 1.9)

### 3.1. Chính sách (Policy)
Chính sách $\pi$ là phân phối xác suất chọn hành động tại trạng thái $s$:
$$\pi(a \mid s) \doteq \mathbb{P}(A_t = a \mid S_t = s)$$

### 3.2. Hai loại Hàm Giá trị
1. **Hàm Giá trị Trạng thái (State-Value Function $v_\pi(s)$)**: Giá trị kỳ vọng khi bắt đầu từ $s$ và tuân theo chính sách $\pi$:
   $$v_\pi(s) \doteq \mathbb{E}_\pi [G_t \mid S_t = s]$$
2. **Hàm Giá trị Hành động (Action-Value Function $q_\pi(s, a)$)**: Giá trị kỳ vọng khi bắt đầu từ $s$, thực hiện hành động $a$, và sau đó tuân theo $\pi$:
   $$q_\pi(s, a) \doteq \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$$

Quan hệ qua lại:
$$v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) q_\pi(s, a)$$

---

### 3.3. Phương trình Bellman (Bellman Expectation Equations)
Biểu diễn mối quan hệ đệ quy giữa giá trị của một trạng thái với giá trị của các trạng thái kế tiếp:

#### Bellman Equation cho $v_\pi(s)$:
$$v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right]$$

#### Bellman Equation cho $q_\pi(s, a)$:
$$q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \sum_{a'} \pi(a' \mid s') q_\pi(s', a') \right]$$

---

### 3.4. Tính Tối ưu & Phương trình Bellman Tối ưu (Bellman Optimality Equations)
Chính sách $\pi_*$ là tối ưu nếu $v_{\pi_*}(s) \ge v_\pi(s), \forall s, \forall \pi$.
Khi đó:
$$v_*(s) \doteq \max_\pi v_\pi(s) = \max_a q_*(s, a)$$
$$q_*(s, a) \doteq \max_\pi q_\pi(s, a)$$

#### Phương trình Bellman Tối ưu cho $v_*(s)$:
$$v_*(s) = \max_{a \in \mathcal{A}} \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_*(s') \right]$$

#### Phương trình Bellman Tối ưu cho $q_*(s, a)$:
$$q_*(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \max_{a'} q_*(s', a') \right]$$

> **Ý nghĩa cực kỳ quan trọng**: Nếu ta biết được $q_*(s, a)$, việc tìm chính sách tối ưu trở nên hiển nhiên bằng cách chọn hành động tham lam: $\pi_*(a \mid s) = 1$ nếu $a = \arg\max_{a'} q_*(s, a')$.

---

## 4. Quy hoạch Động (Dynamic Programming - DP) (Slide 1.10 – 1.12)

Quy hoạch động giả định **biết trước toàn bộ mô hình môi trường** $p(s', r \mid s, a)$.

### 4.1. Đánh giá Chính sách (Policy Evaluation - Prediction)
Tính toán $v_\pi$ từ một chính sách $\pi$ đã cho bằng phép lặp Bellman:
$$V_{k+1}(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) [r + \gamma V_k(s')]$$
Thuật toán dừng khi độ thay đổi lớn nhất $\Delta = \max_s |V_{k+1}(s) - V_k(s)| < \theta$ (với $\theta$ là ngưỡng sai số cực nhỏ).

### 4.2. Cải tiến Chính sách (Policy Improvement)
**Định lý Cải tiến Chính sách (Policy Improvement Theorem)**:  
Nếu với hai chính sách tất định $\pi$ và $\pi'$ thỏa mãn:
$$q_\pi(s, \pi'(s)) \ge v_\pi(s), \quad \forall s \in \mathcal{S}$$
thì chính sách $\pi'$ tốt hơn hoặc bằng $\pi$ trên mọi trạng thái: $v_{\pi'}(s) \ge v_\pi(s)$.

Do đó, ta luôn có thể tạo ra chính sách mới $\pi'$ tốt hơn bằng cách hành động tham lam đối với $v_\pi$:
$$\pi'(s) \doteq \arg\max_a q_\pi(s, a) = \arg\max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_\pi(s')]$$

### 4.3. Thuật toán Lặp Chính sách (Policy Iteration - Control)
Luân phiên giữa Policy Evaluation và Policy Improvement:
$$\pi_0 \xrightarrow{\text{E}} v_{\pi_0} \xrightarrow{\text{I}} \pi_1 \xrightarrow{\text{E}} v_{\pi_1} \xrightarrow{\text{I}} \dots \xrightarrow{\text{I}} \pi_* \xrightarrow{\text{E}} v_*$$
Vì số trạng thái hữu hạn, Policy Iteration đảm bảo hội tụ chính xác về $(\pi_*, v_*)$ sau một số hữu hạn bước lặp!

### 4.4. Lặp Giá trị (Value Iteration)
Nhược điểm của Policy Iteration: Ở mỗi vòng lặp, ta phải chờ Policy Evaluation chạy nhiều bước đến khi hội tụ.  
**Value Iteration** rút ngắn quá trình bằng cách chỉ cập nhật Bellman Optimality 1 lần duy nhất trên mỗi bước:
$$V_{k+1}(s) \leftarrow \max_{a \in \mathcal{A}} \sum_{s', r} p(s', r \mid s, a) [r + \gamma V_k(s')]$$
Dừng khi $\Delta < \theta$, sau đó suy ra chính sách tối ưu:
$$\pi_*(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma V(s')]$$

---

### 4.5. Generalized Policy Iteration (GPI) (Slide 1.12)
GPI là khái niệm tổng quát mô tả sự tương tác cạnh tranh và hợp tác giữa hai quá trình:
1. **Đánh giá (Evaluation)**: Làm cho hàm giá trị $V$ tiệm cận với chính sách hiện tại $\pi$.
2. **Cải tiến (Improvement)**: Làm cho chính sách $\pi$ tham lam hơn đối với hàm giá trị hiện tại $V$.

Hai quá trình này liên tục tương tác, điều chỉnh lẫn nhau cho đến khi tìm được điểm cân bằng duy nhất: điểm tối ưu $(\pi_*, v_*)$.
Toàn bộ các thuật toán RL nâng cao sau này (Monte Carlo, Sarsa, Q-Learning, Deep RL) đều vận hành dựa trên cơ chế GPI!
