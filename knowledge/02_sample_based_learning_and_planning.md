# MODULE 2: PHƯƠNG PHÁP DỰA TRÊN MẪU (MONTE CARLO, TEMPORAL DIFFERENCE) & LẬP KẾ HOẠCH (PLANNING)

> **Tài liệu tương ứng từ bài giảng**: Slide `2.1` đến `2.13`.  
> **Mục tiêu**: Giải quyết bài toán cốt lõi khi **không có sẵn mô hình môi trường** ($p(s', r \mid s, a)$ chưa biết). Tìm hiểu hai trường phái học tăng cường chính: Monte Carlo (MC) và Temporal Difference (TD), cùng kiến trúc kết hợp học trực tiếp với lập kế hoạch (Dyna).

---

## 1. Phương pháp Monte Carlo (MC) (Slide 2.1 – 2.4)

### 1.1. Khái niệm & Nguyên lý Dự đoán (Prediction)
Monte Carlo ước lượng hàm giá trị từ **trải nghiệm thực tế** bằng cách lấy trung bình mẫu của các Lợi nhuận (Sample Returns) $G_t$ thu được sau khi kết thúc cả episode:
$$V(s) \approx \frac{1}{|Episodes|} \sum G_t(s)$$

#### So sánh First-visit MC và Every-visit MC:
- **First-visit MC**: Chỉ ghi nhận lợi nhuận $G_t$ tại lần đầu tiên trạng thái $s$ xuất hiện trong một episode. Cho ước lượng không chệch (unbiased).
- **Every-visit MC**: Ghi nhận lợi nhuận tại tất cả các thời điểm trạng thái $s$ xuất hiện trong episode. Dễ triển khai hơn, hội tụ tiệm cận về giá trị thực.

#### Ưu điểm của MC:
1. **Không cần mô hình môi trường** (Model-free).
2. Giá trị của một trạng thái được ước lượng độc lập, không phụ thuộc vào giá trị của các trạng thái khác (không lan truyền sai số).
3. Có thể tập trung ước lượng chỉ cho một tập con trạng thái quan tâm mà không cần quét toàn bộ không gian trạng thái.

---

### 1.2. Monte Carlo cho Điều khiển (MC for Control) & Vấn đề Khám phá (Slide 2.2 – 2.3)
Vì không có mô hình $p(s', r \mid s, a)$, ta không thể suy ra chính sách tham lam từ $V(s)$. Bắt buộc phải ước lượng **Hàm giá trị hành động $Q(s, a)$**.

#### Vấn đề Khám phá & Khởi tạo Khám phá (Exploring Starts):
- Nếu chính sách $\pi$ là tất định, nhiều cặp $(s, a)$ sẽ không bao giờ được ghé thăm.
- **Exploring Starts (ES)**: Giả định mỗi cặp $(s, a)$ đều có xác suất $> 0$ được chọn làm điểm khởi đầu của episode.
- **Hạn chế của ES**: Phi thực tế trong môi trường thực tế (ví dụ: lái xe tự hành không thể ngẫu nhiên xuất hiện giữa vực sâu).
- **Giải pháp thực tế: $\epsilon$-soft Policy**: Mọi hành động tại mọi trạng thái luôn có xác suất tối thiểu $\frac{\epsilon}{|\mathcal{A}|} > 0$ được chọn.

---

### 1.3. Học Ngoại Chính sách (Off-policy Learning) & Importance Sampling (Slide 2.4)
Trong RL, ta thường đối mặt với xung đột:
- Muốn học được **chính sách tối ưu** (Target Policy $\pi$, thường là tham lam).
- Nhưng phải duy trì hành vi có **tính khám phá cao** (Behavior Policy $b$, ví dụ $\epsilon$-greedy).

#### Tỷ số Trọng số Quan trọng (Importance Sampling Ratio $\rho_{t:T-1}$):
Xác suất của cùng một chuỗi quỹ đạo dưới hai chính sách khác nhau:
$$\rho_{t:T-1} \doteq \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}$$

#### Hai cách tính Importance Sampling:
1. **Ordinary Importance Sampling**:
   $$V(s) \doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{|\mathcal{T}(s)|}$$
   - *Đặc điểm*: Không chệch (unbiased), nhưng **phương sai có thể vô hạn** (rất nguy hiểm khi episode dài).
2. **Weighted Importance Sampling**:
   $$V(s) \doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}}$$
   - *Đặc điểm*: Có chệch nhẹ ban đầu (biased), nhưng **phương sai nhỏ hơn rất nhiều**, ổn định thực tế hơn.

---

## 2. Phương pháp Temporal Difference (TD) (Slide 2.5 – 2.9)

### 2.1. Bản chất của TD(0) & Cơ chế Bootstrapping
Temporal Difference là sự kết hợp tinh hoa giữa Monte Carlo và Quy hoạch động:
- Giống MC: **Học trực tiếp từ trải nghiệm** (không cần mô hình $p$).
- Giống DP: **Tự cập nhật dựa trên ước lượng sẵn có** (gọi là **Bootstrapping**), không cần chờ episode kết thúc.

#### Quy tắc Cập nhật TD(0):
$$V(S_t) \leftarrow V(S_t) + \alpha \big[ \underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD Target}} - V(S_t) \big]$$

#### Sai số Hiệu số Thời gian (TD Error $\delta_t$):
$$\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

---

### 2.2. So sánh Toàn diện: DP vs. MC vs. TD

```
                       [Đầy đủ (Bootstrapping)]
                                 |
                                 v
                              [  DP  ]  (Dùng Model, tính kỳ vọng toán học)
                                 |
 [Chờ kết thúc Episode] ---------+--------- [Cập nhật Tức thời (Online)]
                                 |
                                 v
                              [  TD  ]  (Lấy mẫu thực tế + Bootstrapping)
                                 ^
                                 |
                              [  MC  ]  (Lấy mẫu thực tế, KHÔNG Bootstrapping)
                                 |
                                 v
                       [Mẫu (Sampling)]
```

| Đặc tính | Dynamic Programming (DP) | Monte Carlo (MC) | Temporal Difference (TD) |
| :--- | :--- | :--- | :--- |
| **Yêu cầu Mô hình** | Cần mô hình đầy đủ $p(s', r \mid s, a)$ | Không cần (Model-free) | Không cần (Model-free) |
| **Cách thu thập** | Toàn bộ phân phối (Kỳ vọng) | Trải nghiệm mẫu (Sampling) | Trải nghiệm mẫu (Sampling) |
| **Bootstrapping?** | **CÓ** (dựa vào $V(s')$) | **KHÔNG** (dùng toàn bộ $G_t$) | **CÓ** (dùng $R + \gamma V(S')$) |
| **Thời điểm học** | Quét toàn bộ bảng trạng thái | Phải chờ **hết Episode** | **Online từng bước** ($t \to t+1$) |
| **Tính phù hợp** | Không gian trạng thái nhỏ, biết rõ quy luật | Bài toán episodic | Cả episodic lẫn **continuing tasks** |
| **Phương sai / Độ chệch** | Không chệch, không phương sai | **Không chệch, phương sai cao** | **Có chệch nhẹ, phương sai thấp** |

---

### 2.3. Các thuật toán TD Control (Slide 2.7 – 2.9)

#### 1. SARSA: On-Policy TD Control
- Tên gọi bắt nguồn từ tuple chuyển trạng thái: $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$.
- Cập nhật dựa trên hành động $A_{t+1}$ **thực sự được chọn** bởi chính sách hiện tại (ví dụ $\epsilon$-greedy):
  $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

#### 2. Q-Learning (Watkins 1989): Off-Policy TD Control
- Cập nhật hàm giá trị trực tiếp về $q_*$ bằng cách lấy giá trị cực đại trên mọi hành động khả dĩ ở bước tiếp theo, **bất kể hành động nào được thực thi**:
  $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$
- TD Target của Q-learning: $R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$.

#### 3. Expected SARSA: Giảm phương sai
- Thay vì lấy mẫu $A_{t+1}$ ngẫu nhiên như Sarsa, hoặc lấy cực đại tuyệt đối như Q-Learning, Expected Sarsa tính **kỳ vọng toán học** theo phân phối chính sách $\pi$:
  $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_{a} \pi(a \mid S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$
- **Ưu điểm vượt trội**: Loại bỏ hoàn toàn phương sai do bước chọn hành động tiếp theo gây ra, học mượt mà và ổn định hơn rất nhiều so với Sarsa.
- **Tính linh hoạt**: Có thể chạy On-policy (nếu $\pi$ là chính sách tác nhân đang theo dõi) hoặc Off-policy (nếu $\pi$ khác với behavior policy $b$).

> 📌 **Lưu ý đặc biệt**: Expected Sarsa chính là nền tảng trực tiếp của **Assignment 1** (kết hợp với mạng nơ-ron xấp xỉ và Replay Buffer)!

---

## 3. Lập kế hoạch & Kiến trúc Dyna (Slide 2.10 – 2.13)

### 3.1. Mô hình trong RL (Models)
Mô hình mô phỏng hành vi của môi trường để sinh ra trải nghiệm nhân tạo (simulated experience):
- **Distribution Model**: Sinh ra toàn bộ phân phối xác suất của trạng thái kế tiếp và phần thưởng.
- **Sample Model**: Chỉ sinh ra một mẫu kết quả cụ thể $(S', R)$ ứng với đầu vào $(S, A)$ (như game engine).

### 3.2. Kiến trúc Tabular Dyna-Q (Slide 2.12)
Dyna kết hợp 3 quá trình đồng thời:
1. **Direct RL**: Tác nhân tương tác với môi trường thật để cập nhật $Q(S, A)$ (ví dụ bằng Q-Learning).
2. **Model Learning**: Dùng kinh nghiệm thật để huấn luyện mô hình dự đoán: $\text{Model}(S, A) \to (R, S')$.
3. **Planning**: Lấy ngẫu nhiên $n$ cặp $(S, A)$ đã từng gặp từ bộ nhớ, hỏi $\text{Model}$ để lấy trải nghiệm ảo và tiếp tục cập nhật $Q(S, A)$.

```
                      +-------------------+
                      |    Môi trường     |
                      +-------------------+
                        ^               |
                 Action |               | Real Experience
                        |               v
                   +-------------------------+
       +---------> |    Direct RL Updates    |
       |           +-------------------------+
       |                        |
       |                        v
+-------------+           +-----------+          +-------------------+
|  Chính sách | <-------- |   Q(S, A) | <------- |  Planning Updates |
+-------------+           +-----------+          +-------------------+
                                                        ^
                                                        | Simulated Experience
                                                 +-------------------+
                                                 |   Model Môi trường|
                                                 +-------------------+
                                                        ^
                                                        | Model Learning
                                           Kinh nghiệm thật (S, A, R, S')
```

### 3.3. Xử lý Môi trường Biến đổi (Dyna-Q+) (Slide 2.13)
- Khi môi trường thay đổi (ví dụ đường đi bị chặn - *blocking maze*, hoặc đường tắt mới mở ra - *shortcut maze*), mô hình đã học sẽ bị sai lệch.
- **Dyna-Q+**: Thêm phần thưởng khuyến khích tò mò (Exploration Bonus):
  $$R_{\text{bonus}} = R + \kappa \sqrt{\tau}$$
  với $\tau$ là số bước thời gian kể từ lần cuối cùng cặp $(S, A)$ được thử nghiệm trong thực tế, $\kappa$ là hệ số trọng số. Càng lâu chưa được ghé thăm, phần thưởng ảo càng lớn, kích thích tác nhân tái kiểm tra các lối đi cũ để phát hiện đường tắt mới!
