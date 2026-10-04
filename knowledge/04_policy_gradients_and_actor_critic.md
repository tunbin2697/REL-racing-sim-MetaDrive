# MODULE 3B: BÀI TOÁN CONTINUING, ĐỊNH LÝ POLICY GRADIENT & KIẾN TRÚC ACTOR-CRITIC

> **Tài liệu tương ứng từ bài giảng**: Slide `3.8` đến `3.12`.  
> **Trọng tâm thực hành**: Nền tảng toán học và thuật toán trực tiếp của **Assignment 3** (Pendulum Swing-Up với Average Reward Softmax Actor-Critic).

---

## 1. Bài toán Continuing Tasks & Công thức Phần thưởng Trung bình (Average Reward) (Slide 3.8)

### 1.1. Tại sao Discount Factor ($\gamma < 1$) lại Thất bại trong Continuing Tasks với Function Approximation?
Trong bài toán không hồi kết (Continuing Tasks):
- Nếu sử dụng hệ số chiết khấu $\gamma < 1$, giá trị của một trạng thái $v_\gamma(s)$ phụ thuộc nặng nề vào việc trạng thái đó xuất hiện **sớm hay muộn** trong chuỗi thời gian, chứ không phản ánh đúng hiệu quả lâu dài của chính sách.
- Khi kết hợp với xấp xỉ hàm (Function Approximation), việc chiết khấu tạo ra xung đột nghiêm trọng giữa việc tối ưu hóa mục tiêu tại các bước đầu và các bước sau, dẫn đến việc chính sách tối ưu tìm được không thực sự tối đa hóa hiệu năng trung bình dài hạn!

👉 **Giải pháp**: Sử dụng thiết lập **Phần thưởng Trung bình (Average Reward Formulation)** với $\gamma = 1$.

---

### 1.2. Định nghĩa Tỷ lệ Phần thưởng Trung bình $r(\pi)$
Hiệu năng của một chính sách $\pi$ được định nghĩa là tỷ lệ phần thưởng kỳ vọng trên mỗi bước thời gian khi tác nhân hoạt động vô hạn:
$$r(\pi) \doteq \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T \mathbb{E}[R_t \mid S_0, A_{0:t-1} \sim \pi] = \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) r$$
với $d^\pi(s) \doteq \lim_{t \to \infty} \mathbb{P}(S_t = s \mid A_{0:t-1} \sim \pi)$ là **phân phối trạng thái dừng (Stationary Distribution)**.

### 1.3. Lợi nhuận Vi phân (Differential Return) & Hàm Giá trị Vi phân
Vì $T \to \infty$, tổng phần thưởng thông thường sẽ phân kỳ ($\to \infty$). Để giải quyết, ta trừ đi phần thưởng trung bình $r(\pi)$ tại mỗi bước thời gian.

#### Lợi nhuận Vi phân (Differential Return $G_t$):
$$G_t \doteq (R_{t+1} - r(\pi)) + (R_{t+2} - r(\pi)) + (R_{t+3} - r(\pi)) + \dots = \sum_{k=0}^\infty (R_{t+k+1} - r(\pi))$$

#### Hàm Giá trị Trạng thái Vi phân (Differential State-Value $v_\pi(s)$):
$$v_\pi(s) \doteq \mathbb{E}_\pi [G_t \mid S_t = s]$$

#### Phương trình Bellman Vi phân cho $v_\pi$:
$$v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r - r(\pi) + v_\pi(s') \right]$$

---

## 2. Học Chính sách Tham số hóa & Định lý Policy Gradient (Slide 3.9 – 3.10)

### 2.1. Tại sao lại Tối ưu hóa Trực tiếp Chính sách $\pi_\theta$?
Thay vì học hàm giá trị $Q(s, a)$ rồi mới suy ra hành động (Value-based), ta tham số hóa chính sách bằng vector $\theta \in \mathbb{R}^{d'}$:
$$\pi(a \mid s, \theta) = \mathbb{P}(A_t = a \mid S_t = s, \theta)$$

#### Ưu điểm vượt trội:
1. **Học được chính sách ngẫu nhiên (Stochastic Policies)**: Trong các bài toán thông tin không hoàn hảo (POMDPs) hoặc game đối kháng (oẳn tù tì), chính sách tối ưu bắt buộc phải là xác suất. Value-based (tham lam) sẽ bị kẹt.
2. **Xử lý mượt mà không gian hành động liên tục**: Không cần làm phép tính $\max_a Q(s, a)$ vốn cực kỳ đắt đỏ khi $a \in \mathbb{R}^n$.
3. **Sự thay đổi chính sách diễn ra liên tục, mượt mà**: Tránh hiện tượng thay đổi đột ngột chính sách (chattering) khi hai giá trị $Q(s, a_1) \approx Q(s, a_2)$.

---

### 2.2. Định lý Policy Gradient cho Continuing Tasks (Slide 3.10)
Mục tiêu là tối đa hóa hiệu năng trung bình:
$$J(\theta) \doteq r(\pi_\theta)$$

**Định lý Policy Gradient (Sutton et al., 1999)**:  
Đạo hàm của hàm mục tiêu $J(\theta)$ theo vector tham số $\theta$ được tính chính xác bằng:
$$\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} q_\pi(s, a) \nabla_\theta \pi(a \mid s, \theta)$$
Sử dụng mẹo Log-Derivative ($\nabla_\theta \pi = \pi \nabla_\theta \ln \pi$):
$$\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ q_\pi(S_t, A_t) \nabla_\theta \ln \pi(A_t \mid S_t, \theta) \right]$$

> 💎 **Ý nghĩa kỳ diệu của định lý**: Đạo hàm của $J(\theta)$ **hoàn toàn không phụ thuộc vào đạo hàm của phân phối trạng thái dừng $d^\pi(s)$ theo $\theta$**! Điều này cho phép ta lấy mẫu ngẫu nhiên từ quỹ đạo chạy thực tế mà vẫn thu được đạo hàm chính xác!

---

## 3. Kiến trúc Actor-Critic cho Continuing Tasks (Slide 3.11)

### 3.1. Phân vai: Actor và Critic
- **Critic (Nhà phê bình)**: Ước lượng hàm giá trị vi phân $v(s, w) \approx v_\pi(s)$.
- **Actor (Diễn viên)**: Điều chỉnh tham số chính sách $\theta$ dựa trên sự đánh giá của Critic.

```
                  +--------------------------------+
                  |         Môi trường             |
                  +--------------------------------+
                     ^                          |
            Hành động|                          | Trạng thái S, Phần thưởng R
            A_t      |                          v
                +---------+             +----------------+
                |  ACTOR  |             |     CRITIC     |
                |  pi_theta |           |    v(s, w)     |
                +---------+             +----------------+
                     ^                          |
                     | Cập nhật theta           | Đánh giá v(S), v(S')
                     +----------- Sai số -------+
                                  TD Error delta
```

---

### 3.2. Sai số TD Vi phân (Differential TD Error $\delta_t$)
Trong bài toán Continuing với phần thưởng trung bình $\bar{R}$, sai số TD là:
$$\delta_t \doteq R_{t+1} - \bar{R}_t + \hat{v}(S_{t+1}, w_t) - \hat{v}(S_t, w_t)$$

### 3.3. Bộ 3 Quy tắc Cập nhật Đồng thời:
Tại mỗi bước thời gian $t$, tác nhân cập nhật đồng thời cả 3 biến số:

1. **Cập nhật Ước lượng Phần thưởng Trung bình $\bar{R}$**:
   $$\bar{R}_{t+1} \leftarrow \bar{R}_t + \alpha_r \delta_t$$
2. **Cập nhật Trọng số Critic $w$**:
   $$w_{t+1} \leftarrow w_t + \alpha_w \delta_t \nabla_w \hat{v}(S_t, w_t)$$
   *(Nếu dùng xấp xỉ tuyến tính $\hat{v}(s, w) = w^T x(s)$, thì $\nabla_w \hat{v} = x(S_t)$)*.
3. **Cập nhật Tham số Actor $\theta$**:
   $$\theta_{t+1} \leftarrow \theta_t + \alpha_\theta \delta_t \nabla_\theta \ln \pi(A_t \mid S_t, \theta_t)$$

---

## 4. Tham số hóa Chính sách Softmax Tuyến tính (Slide 3.12)

### 4.1. Độ ưu tiên Hành động (Action Preferences)
Với không gian hành động rời rạc $\mathcal{A}$, ta gán cho mỗi hành động một độ ưu tiên tuyến tính $h(s, a, \theta)$:
$$h(s, a, \theta) \doteq \theta^T x(s, a)$$

Chính sách Softmax:
$$\pi(a \mid s, \theta) \doteq \frac{e^{h(s, a, \theta)}}{\sum_{b \in \mathcal{A}} e^{h(s, b, \theta)}}$$

### 4.2. Đạo hàm Log-Policy ($\nabla_\theta \ln \pi$)
Đạo hàm giải tích của logarit chính sách Softmax có dạng cực kỳ đẹp mắt:
$$\nabla_\theta \ln \pi(a \mid s, \theta) = x(s, a) - \sum_{b \in \mathcal{A}} \pi(b \mid s, \theta) x(s, b)$$

#### Biểu diễn Dưới dạng Ma trận Trọng số $\theta$:
Nếu ta tổ chức $\theta$ thành ma trận kích thước $(|\mathcal{A}|, d)$ (mỗi hàng ứng với một hành động), và tại trạng thái $s$ có tập các active tiles $x(s)$:
- Độ ưu tiên: $h(s, a) = \sum_{i \in \text{tiles}} \theta[a, i]$.
- Với hành động được chọn $A_t$:
  $$\nabla_{\theta[A_t, :]} \ln \pi = (1 - \pi(A_t \mid S_t)) x(S_t)$$
- Với mọi hành động khác $a \neq A_t$:
  $$\nabla_{\theta[a, :]} \ln \pi = -\pi(a \mid S_t) x(S_t)$$

Quy tắc cập nhật trọng số Actor cho toàn bộ các hành động $a \in \mathcal{A}$:
$$\theta[a, \text{tiles}] \leftarrow \theta[a, \text{tiles}] + \alpha_\theta \delta_t \left[ \mathbb{I}_{a = A_t} - \pi(a \mid S_t) \right]$$

> 🚀 **Trọng tâm Assignment 3**: Công thức trên chính là chìa khóa then chốt để lập trình hàm `agent_step` trong **Assignment 3 (Pendulum Swing-Up)**!
