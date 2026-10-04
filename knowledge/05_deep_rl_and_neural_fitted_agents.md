# CHUYÊN ĐỀ: DEEP RL, BỘ TỐI ƯU ADAM & NEURAL FITTED AGENTS

> **Tài liệu tương ứng**: Slide `3.1` (Mở rộng Deep RL) & **Assignment 1** (Deep Expected Sarsa với Replay Buffer & Adam trên môi trường Lunar Lander).  
> **Mục tiêu**: Nắm trọn vẹn kiến trúc kết hợp Mạng nơ-ron sâu (Deep Neural Networks) vào Học tăng cường để giải quyết bài toán không gian trạng thái liên tục phức tạp.

---

## 1. Thách thức "Tam giác Chết chóc" (The Deadly Triad)

Trong bài báo kinh điển của Tsitsiklis & Van Roy (1997), việc kết hợp 3 yếu tố sau sẽ tiềm ẩn nguy cơ **bùng nổ giá trị (divergence)** hoặc mất ổn định nghiêm trọng:
1. **Xấp xỉ hàm (Function Approximation)**: Sử dụng mô hình phi tuyến tính phức tạp như Mạng Nơ-ron.
2. **Tự ước lượng (Bootstrapping)**: Dùng ước lượng trạng thái sau để cập nhật trạng thái trước (như trong TD, Sarsa, Q-learning).
3. **Học ngoại chính sách (Off-policy)**: Phân phối dữ liệu huấn luyện khác với phân phối của chính sách đang được tối ưu hóa.

Để Deep RL có thể huấn luyện thành công và ổn định trên các bài toán như Lunar Lander hay Atari, ta bắt buộc phải sử dụng 3 "vũ khí tối thượng":
- **Experience Replay Buffer** (Phá vỡ tương quan thời gian).
- **Target Network / Fixed Q-Network** (Cố định mục tiêu học).
- **Bộ tối ưu hóa thích nghi Adam** (Ổn định bước nhảy gradient).

---

## 2. Mạng Nơ-ron Giá trị Hành động (Action-Value Network)

### 2.1. Cấu trúc Mạng MLP (Multi-Layer Perceptron)
Thay vì đưa cả cặp $(s, a)$ vào đầu vào, mạng nơ-ron trong Q-Learning/Expected Sarsa nhận vector trạng thái $s \in \mathbb{R}^{d_{\text{in}}}$ làm đầu vào và xuất ra vector giá trị của **tất cả các hành động cùng lúc**:
$$\mathbf{q}(s; \mathbf{W}) = \big[ Q(s, a_1; \mathbf{W}), Q(s, a_2; \mathbf{W}), \dots, Q(s, a_k; \mathbf{W}) \big]^T$$

```
  [Input Layer: State s]   --->   [Hidden Layer 1 (ReLU)]   --->   [Output Layer: Linear]
         (d_state)                       (num_hidden)                    (num_actions)
                                                                       Q(s, a_1)
            o                             O -----\                     Q(s, a_2)
            o            =======>         O ======>                    Q(s, a_3)
            o                             O -----/                     Q(s, a_4)
```

- **Hàm kích hoạt tầng ẩn**: Thường là **ReLU** ($g(z) = \max(0, z)$) giúp tránh hiện tượng biến mất gradient (vanishing gradient).
- **Tầng đầu ra**: **Linear** (không dùng hàm kích hoạt phi tuyến) vì giá trị $Q(s, a)$ có thể nhận bất kỳ giá trị số thực âm hay dương nào!

### 2.2. Lan truyền Xuôi (Forward Pass) & Lan truyền Ngược (Backpropagation)
- Với mỗi tầng $l$:
  $$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
  $$a^{[l]} = g(z^{[l]})$$
- Khi cập nhật Bán Gradient (Semi-gradient), ta coi TD error $\delta$ là tín hiệu sai số tại tầng ra ứng với hành động đã thực hiện, sau đó truyền ngược đạo hàm về các tầng trước để tính $\nabla_{W^{[l]}}$ và $\nabla_{b^{[l]}}$.

---

## 3. Bộ nhớ đệm Trải nghiệm (Experience Replay Buffer)

### 3.1. Vấn đề Dữ liệu Tương quan Tuần tự (Temporal Autocorrelation)
Trong RL trực tuyến truyền thống, các trạng thái xuất hiện kế tiếp nhau $S_t, S_{t+1}, S_{t+2}$ phụ thuộc rất chặt chẽ vào nhau theo thời gian:
- Khi mạng nơ-ron học liên tục trên các mẫu liền kề, nó sẽ bị cuốn theo phân phối cục bộ và nhanh chóng "quên sạch" các kỹ năng đã học ở vùng trạng thái khác (**Catastrophic Forgetting**).
- Gradient cập nhật bị dao động cực mạnh, không thỏa mãn giả định dữ liệu độc lập và cùng phân phối ($i.i.d.$) của học máy.

### 3.2. Cơ chế Hoạt động của Replay Buffer
1. **Lưu trữ**: Tại mỗi bước $t$, lưu transition $(S_t, A_t, R_{t+1}, \text{terminal}, S_{t+1})$ vào một bộ nhớ đệm tròn (Circular FIFO Buffer) có dung lượng cố định (ví dụ $50,000$ mẫu).
2. **Lấy mẫu Ngẫu nhiên Đồng đều (Uniform Random Sampling)**:
   - Khi cập nhật mạng, bốc ngẫu nhiên một minibatch gồm $B$ mẫu độc lập từ buffer (ví dụ $B = 64$).
   - Các mẫu trong minibatch đến từ nhiều thời điểm và episode hoàn toàn khác nhau $\to$ **Phá vỡ hoàn toàn tương quan chuỗi thời gian**, biến bài toán RL về dạng huấn luyện ổn định tương tự học có giám sát!

---

## 4. Bộ Tối ưu hóa Adam (Adaptive Moment Estimation)

Thuật toán Adam (Kingma & Ba, 2014) là bộ tối ưu hóa tiêu chuẩn vàng trong Deep Learning và Deep RL.

### 4.1. Toán học của Adam
Tại bước lặp $t$, với gradient $g_t$:
1. **Moment bậc nhất (Ước lượng Kỳ vọng Gradient - Momentum)**:
   $$m_t = \beta_m m_{t-1} + (1 - \beta_m) g_t$$
   *(Thông thường $\beta_m = 0.9$)*.
2. **Moment bậc hai (Ước lượng Phương sai không định tâm - RMSprop)**:
   $$v_t = \beta_v v_{t-1} + (1 - \beta_v) g_t^2$$
   *(Thông thường $\beta_v = 0.999$)*.
3. **Hiệu chỉnh Độ chệch (Bias Correction)**:
   Vì $m_0 = 0$ và $v_0 = 0$, trong các bước đầu $m_t, v_t$ bị lệch về 0 rất nặng. Adam hiệu chỉnh bằng cách chia cho $(1 - \beta^t)$:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_m^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_v^t}$$
4. **Cập nhật Trọng số**:
   $$W_t = W_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon_{\text{adam}}} \hat{m}_t$$

### 4.2. Lưu ý Lập trình Thực tế: Tính Lũy thừa Tăng dần
Để tránh tính $\beta^t$ bằng hàm số mũ đắt đỏ hoặc bị tràn số, ta cập nhật tích số tích lũy:
```python
self.beta_m_product *= self.beta_m
self.beta_v_product *= self.beta_v
```
Khi đó mẫu số hiệu chỉnh đơn giản là `1.0 - self.beta_m_product`.

---

## 5. Chính sách Softmax & Kỹ thuật Trừ Max Chống Tràn số (Log-Sum-Exp Trick)

### 5.1. Định nghĩa Chính sách Softmax với Nhiệt độ $\tau$
Xác suất chọn hành động $a$ tỷ lệ thuận với giá trị số mũ của $Q(s, a)$ chia cho tham số nhiệt độ $\tau > 0$:
$$\pi(a \mid s) = \frac{\exp\left(\frac{Q(s, a)}{\tau}\right)}{\sum_{b \in \mathcal{A}} \exp\left(\frac{Q(s, b)}{\tau}\right)}$$

- Khi $\tau \to \infty$: Tất cả hành động có xác suất bằng nhau $\frac{1}{|\mathcal{A}|}$ (Khám phá tối đa).
- Khi $\tau \to 0$: Hành động có $Q$ cao nhất chiếm xác suất $\to 1$ (Khai thác tuyệt đối / Greedy).

### 5.2. Hiểm họa Tràn số Float Overflow & Kỹ thuật Trừ Max
- Hàm số mũ $e^x$ tăng theo cấp số nhân cực nhanh. Trong Python/NumPy, nếu $x > 709.78$, `np.exp(x)` sẽ lập tức biến thành `inf`, dẫn đến phép chia $\frac{\infty}{\infty} \to \text{NaN}$!
- **Kỹ thuật Trừ Cực đại (Max Subtraction / Log-Sum-Exp Trick)**:  
  Dựa trên tính chất đại số:
  $$\frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{-c} \cdot e^{z_i}}{e^{-c} \cdot \sum_j e^{z_j}} = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}}$$
  Chọn $c = \max_j z_j$:
  - Số mũ lớn nhất sẽ là $z_{\max} - c = 0 \implies e^0 = 1$.
  - Mọi số mũ khác đều là số âm ($z_i - c \le 0 \implies e^{z_i - c} \in (0, 1]$).
  - Mẫu số luôn luôn $\ge 1$.
  👉 **Triệt tiêu 100% nguy cơ tràn số `inf` hoặc `NaN`!**

---

## 6. Mạng Mục tiêu (Target Network) & TD Error của Expected Sarsa

### 6.1. Vấn đề Mục tiêu Di động (Moving Target Problem)
Nếu ta dùng cùng một mạng $Q(s, a; W)$ để vừa tính giá trị dự đoán $Q(S_t, A_t)$, vừa tính giá trị mục tiêu $R + \gamma \sum \pi(a|S') Q(S', a)$, thì mỗi lần cập nhật trọng số $W$, mục tiêu học cũng bị dịch chuyển theo! Hiện tượng này giống như việc đuổi theo một cái bóng di động, gây ra dao động không thể kiểm soát.

👉 **Giải pháp**: Cố định mạng tính mục tiêu! Ta duy trì một bản sao `current_q` với trọng số đóng băng để tính giá trị ở trạng thái tiếp theo $S_{t+1}$.

### 6.2. Công thức TD Error của Expected Sarsa trong Minibatch
Với mỗi transition $(S_i, A_i, R_i, \text{terminal}_i, S'_i)$ trong minibatch:
1. Tính ma trận $Q$ tại trạng thái kế tiếp từ mạng cố định `current_q`:
   $$Q_{\text{next}} = \text{current\_q.get\_action\_values}(S')$$
2. Tính phân phối chính sách tại $S'$ thông qua hàm `softmax(Q_next, tau)`:
   $$\Pi_{\text{next}} = \text{softmax}(Q_{\text{next}}, \tau)$$
3. Tính giá trị kỳ vọng của trạng thái kế tiếp $V(S')$:
   $$V_{\text{next}}[i] = \sum_{a} \Pi_{\text{next}}[i, a] \cdot Q_{\text{next}}[i, a]$$
4. **Xử lý Trạng thái Kết thúc (Terminal State)**:
   Nếu episode kết thúc tại $S'_i$, thì không có phần thưởng tương lai nào nữa:
   $$V_{\text{next}}[i] = 0 \quad \text{nếu } \text{terminal}_i == 1$$
   *(Biểu thức toán học ngắn gọn: $V_{\text{next}} = (1 - \text{terminals}) \odot V_{\text{next}}$)*.
5. Tính TD Target và TD Error:
   $$\text{Target}_i = R_i + \gamma \cdot V_{\text{next}}[i]$$
   $$\delta_i = \text{Target}_i - Q(S_i, A_i; \text{network})$$

> 🏆 **Tác vụ Assignment 1**: Toàn bộ logic trên được gói gọn trong hàm `get_td_error(...)` và `optimize_network(...)` để giải quyết môi trường Lunar Lander!
