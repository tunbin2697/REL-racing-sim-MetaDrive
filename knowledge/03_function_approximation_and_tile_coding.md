# MODULE 3A: XẤP XỈ HÀM (FUNCTION APPROXIMATION) & KỸ THUẬT TILE CODING

> **Tài liệu tương ứng từ bài giảng**: Slide `3.1` đến `3.7`.  
> **Trọng tâm thực hành**: Nền tảng cốt lõi của **Assignment 2** (Mountain Car với Tile Coding & Sarsa).

---

## 1. Tại sao cần Xấp xỉ Hàm (Function Approximation)? (Slide 3.1 – 3.2)

### 1.1. Lời nguyền Chiều dữ liệu (Curse of Dimensionality)
Trong các bài toán thực tế:
- Không gian trạng thái có thể là **liên tục** (như vị trí, góc quay, vận tốc trong điều khiển robot).
- Hoặc số trạng thái rời rạc quá khổng lồ (như cờ vây với $10^{170}$ thế cờ).
- Ta không thể lưu trữ một bảng lookup $V(s)$ hay $Q(s, a)$. Quan trọng hơn: **tác nhân sẽ không bao giờ gặp lại chính xác cùng một trạng thái hai lần!**

👉 **Giải pháp**: Phải có khả năng **khái quát hóa (Generalization)**: học từ một số trạng thái đã gặp và suy luận ra giá trị cho các trạng thái tương tự chưa từng gặp.

---

### 1.2. Góc nhìn Học có Giám sát (Supervised Learning View)
Ta biểu diễn hàm giá trị dưới dạng một hàm tham số hóa với vector trọng số $w \in \mathbb{R}^d$ ($d \ll |\mathcal{S}|$):
$$\hat{v}(s, w) \approx v_\pi(s)$$

#### Hàm Mất mát Sai số Giá trị Bình phương Trung bình ($\overline{\text{VE}}$):
$$\overline{\text{VE}}(w) \doteq \sum_{s \in \mathcal{S}} \mu(s) \left[ v_\pi(s) - \hat{v}(s, w) \right]^2$$
Trong đó:
- $\mu(s) \ge 0, \sum_s \mu(s) = 1$ là **phân phối trạng thái theo chính sách (On-policy State Distribution)**: đánh trọng số cao hơn cho các trạng thái mà tác nhân ghé thăm thường xuyên.

#### Đạo hàm Giảm Gradient (Gradient Descent):
Nếu $U_t$ là một ước lượng không chệch của $v_\pi(S_t)$ (như trong Monte Carlo $U_t = G_t$):
$$w_{t+1} \leftarrow w_t - \frac{1}{2} \alpha \nabla \left[ U_t - \hat{v}(S_t, w_t) \right]^2 = w_t + \alpha \left[ U_t - \hat{v}(S_t, w_t) \right] \nabla \hat{v}(S_t, w_t)$$

---

## 2. Phương pháp Bán Gradient (Semi-Gradient Methods) (Slide 3.3 – 3.4)

### 2.1. Nghịch lý TD Target & Khái niệm "Semi-gradient"
Trong Temporal Difference, mục tiêu học tập là **TD Target**:
$$U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, w_t)$$
Nhận xét: $U_t$ **bản thân nó cũng phụ thuộc vào vector trọng số $w_t$!**
- Nếu lấy đạo hàm thực sự theo $w$, ta sẽ phải đạo hàm cả số hạng $R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$ bên trong Target. Điều này làm thuật toán cực kỳ phức tạp và không đảm bảo hội tụ.
- **Phương pháp Bán Gradient (Semi-gradient)**: Bỏ qua đạo hàm của Target, coi Target là một hằng số số học tại thời điểm bước $t$:
  $$w_{t+1} \leftarrow w_t + \alpha \left[ R_{t+1} + \gamma \hat{v}(S_{t+1}, w_t) - \hat{v}(S_t, w_t) \right] \nabla \hat{v}(S_t, w_t)$$

---

### 2.2. Xấp xỉ Tuyến tính (Linear Function Approximation) (Slide 3.4)
Trong mô hình tuyến tính, hàm giá trị là tích vô hướng giữa vector trọng số $w$ và vector đặc trưng (feature vector) $x(s) \in \mathbb{R}^d$:
$$\hat{v}(s, w) \doteq w^T x(s) = \sum_{i=1}^d w_i x_i(s)$$

Đạo hàm của hàm tuyến tính theo $w$ cực kỳ đơn giản:
$$\nabla \hat{v}(s, w) = x(s)$$

#### Công thức Cập nhật Linear Semi-gradient TD(0):
$$w_{t+1} \leftarrow w_t + \alpha \delta_t x(S_t)$$
với $\delta_t = R_{t+1} + \gamma w_t^T x(S_{t+1}) - w_t^T x(S_t)$.

> 💡 **Nhận xét sâu sắc**: Phương pháp bảng (Tabular TD) thực chất chỉ là một trường hợp con đặc biệt của Linear TD, trong đó vector $x(s)$ là vector **one-hot** (chỉ có đúng một phần tử mang giá trị 1 ứng với trạng thái $s$, các phần tử còn lại bằng 0).

---

## 3. Kỹ thuật Xây dựng Đặc trưng & Tile Coding (Slide 3.5)

Để biểu diễn các biến số liên tục (như vị trí $x$, vận tốc $v$), ta cần chuyển đổi chúng thành vector đặc trưng $x(s)$.

### 3.1. Phân loại Kỹ thuật Biểu diễn
1. **State Aggregation (Gom cụm trạng thái)**: Chia không gian liên tục thành các khối hộp rời rạc không chồng lấn. Nhược điểm: Giá trị bị đứt gãy tại ranh giới, không có khái quát hóa giữa hai ô kề nhau.
2. **Coarse Coding (Mã hóa thô)**: Sử dụng các miền cảm nhận (receptive fields - như các vòng tròn hoặc hình elip) **chồng lấn lên nhau**. Trạng thái $s$ rơi vào vòng tròn nào thì đặc trưng của vòng tròn đó được kích hoạt ($= 1$).
3. **Tile Coding**: Dạng hiệu quả và chuẩn mực nhất của Coarse Coding cho không gian nhiều chiều!

---

### 3.2. Cấu trúc của Tile Coding
Một bộ mã hóa Tile Coder bao gồm nhiều lưới phân hoạch gọi là **Tilings** (lớp ngói). Mỗi tiling được tạo thành từ nhiều ô con gọi là **Tiles** (viên ngói).

```
Tiling 1 (Lưới gốc)         Tiling 2 (Dịch chuyển nhẹ)
+-----+-----+-----+         +-----+-----+-----+
|     |     |     |       +-----+-----+-----+ |
+-----+--*--+-----+  -->  |     |   * |     | |
|     |     |     |       +-----+-----+-----+ +
+-----+-----+-----+       |     |     |     |
                          +-----+-----+-----+
```

#### Nguyên lý Hoạt động:
1. **Dịch chuyển bất đối xứng (Offsetting)**: Mỗi tiling được dịch chuyển một khoảng vi mô so với tiling trước đó:
   $$\text{offset}_i = \frac{2i - 1}{2k}$$
   với $k$ là số chiều.
2. **Kích hoạt Thưa (Sparse Active Features)**:
   - Tại bất kỳ trạng thái liên tục $s$ nào, trong mỗi tiling sẽ có **chính xác đúng 1 tile chứa điểm $s$**.
   - Do đó, nếu có $m$ tilings, luôn có **chính xác $m$ đặc trưng được kích hoạt ($= 1$)**, tất cả các đặc trưng còn lại trong tổng thể đều bằng $0$!
3. **Ưu thế Tính toán Tuyệt đối**:
   Vì chỉ có $m$ đặc trưng bằng 1, tích vô hướng $w^T x(s)$ không cần nhân ma trận, mà chỉ là **tổng của $m$ trọng số**:
   $$\hat{v}(s, w) = \sum_{i \in \text{active\_tiles}(s)} w_i$$
   Độ phức tạp tính toán là $O(m)$ thay vì phụ thuộc vào kích thước toàn bộ không gian!

### 3.3. Index Hash Table (`IHT`)
Nếu không gian trạng thái lớn (ví dụ $8 \times 8 \times 8 \times \dots$), tổng số tiles lý thuyết có thể lên đến hàng triệu.  
Thư viện `tile_coding` sử dụng bảng băm **Index Hash Table (`IHT`)** kích thước cố định (ví dụ $4096$):
- Sử dụng hàm băm tự động ánh xạ tọa độ lưới $(x, y, \text{tiling\_id}) \to [0, \text{iht\_size} - 1]$.
- Giới hạn bộ nhớ cố định, xử lý va chạm thông minh, cực kỳ nhanh.

### 3.4. Hệ số Học Quy đổi ($\alpha$) trong Tile Coding
Vì tại mỗi bước có $m$ tilings kích hoạt cùng lúc, nếu chọn $\alpha$ thông thường, độ cập nhật sẽ bị nhân lên $m$ lần!  
Do đó, quy tắc chuẩn tắc là đặt **step size chia cho số tilings**:
$$\alpha_{\text{actual}} = \frac{\alpha}{m}$$
Khi đó, $\alpha = 1.0$ tương ứng với việc sau 1 bước cập nhật, sai số được triệt tiêu hoàn toàn trên trạng thái đó!

---

## 4. Điều khiển với Xấp xỉ Hàm: Episodic Semi-gradient Sarsa (Slide 3.6 – 3.7)

### 4.1. Biểu diễn Đặc trưng Phụ thuộc Hành động $x(s, a)$
Trong bài toán điều khiển, ta cần xấp xỉ hàm giá trị hành động:
$$\hat{q}(s, a, w) = w^T x(s, a)$$

#### Cách Xếp chồng Đặc trưng (Feature Stacking) cho các hành động rời rạc:
Nếu có $|\mathcal{A}|$ hành động và vector trạng thái kích hoạt $m$ tiles:
- **Cách 1: Vector xếp chồng (Stacked Vector)**: Vector đặc trưng có kích thước $|\mathcal{A}| \times \text{iht\_size}$. Các tiles của hành động $a$ được tịnh tiến chỉ số:
  $$\text{active\_indices}(s, a) = \text{tiles}(s) + a \times \text{iht\_size}$$
- **Cách 2: Ma trận trọng số 2 chiều**: $w$ có kích thước $(|\mathcal{A}|, \text{iht\_size})$. Giá trị hành động:
  $$\hat{q}(s, a, w) = \sum_{i \in \text{tiles}(s)} w[a, i]$$

### 4.2. Thuật toán Episodic Semi-gradient Sarsa cho Điều khiển
```
Khởi tạo trọng số w tùy ý (ví dụ w = 0)
Lặp lại với mỗi episode:
    S = env.reset()
    tiles = get_tiles(S)
    Tính Q(S, a) = sum(w[a, tiles]) với mọi a
    Chọn A theo epsilon-greedy dựa trên Q(S, :)
    
    Lặp lại cho mỗi bước của episode:
        Thực thi hành động A, quan sát R, S'
        Nếu S' là Terminal:
            w[A, tiles] += alpha * (R - sum(w[A, tiles]))
            Kết thúc episode
            
        next_tiles = get_tiles(S')
        Tính Q(S', a') = sum(w[a', next_tiles]) với mọi a'
        Chọn A' theo epsilon-greedy dựa trên Q(S', :)
        
        delta = R + gamma * Q(S', A') - Q(S, A)
        w[A, tiles] += alpha * delta
        
        S = S', A = A', tiles = next_tiles
```

> 🎯 **Ứng dụng trực tiếp**: Đây chính là thuật toán sinh viên phải lập trình trong **Assignment 2** để giải bài toán xe leo dốc Mountain Car!
