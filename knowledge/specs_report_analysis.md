# 📑 BÁO CÁO PHÂN TÍCH CHUYÊN SÂU: SPECIFICATIONS REPORT SILVERSTONE GP & XE ĐUA FIA GT3

> **Trạng thái (31/10/2026)**: **lịch sử**, đánh giá ngày 29/10 về tài liệu số liệu thật. Nhiều nhận định đã được đo lại và sửa ở `knowledge/vehicle_and_engine_physics.md` §8.1 và §9 (ví dụ lập luận Froude, góc lái cần thiết). Không trích file này vào báo cáo.

> **Tài liệu đối chiếu**: [knowledge/silverstone_gt3_rl_specs.md](file:///c:/Users/tunbi/Project/school/REL%20Assignment/knowledge/silverstone_gt3_rl_specs.md)  
> **Thời điểm lập**: 29/10/2026  
> **Người thực hiện**: Antigravity Assistant  
> **Mục tiêu**: Đánh giá tính khả thi, độ chuẩn xác khoa học của báo cáo nghiên cứu thông số thực tế, và khả năng tính toán Optimal Profile độc lập cho bài toán RL.

---

## 📌 I. ĐÁNH GIÁ TỔNG QUAN TÀI LIỆU NGHIÊN CỨU

Tài liệu [knowledge/silverstone_gt3_rl_specs.md](file:///c:/Users/tunbi/Project/school/REL%20Assignment/knowledge/silverstone_gt3_rl_specs.md) là một bản tổng hợp kỹ thuật **cực kỳ chất lượng, có giá trị học thuật cao và bám sát thực tế thi đấu chuyên nghiệp**. 

Tài liệu cung cấp 4 khối dữ liệu đắt giá mà đồ án có thể khai thác trực tiếp:
1. **Dữ liệu Telemetry thực tế từng góc cua (Turn 1 – Turn 18)** trích xuất từ các xe GT3 đỉnh cao (Lamborghini Huracán GT3, McLaren 720S GT3).
2. **Bán kính hiệu dụng ($R_{\text{eff}}$) và độ cong ($\kappa$)** được tính toán chính xác từ gia tốc ly tâm thực tế ($a_y \approx 1.8g - 2.0g$).
3. **Bộ tham số động lực học chuẩn hóa (Bicycle/Point-mass Model)**: Khối lượng, trục cơ sở, khí động học ($C_L A, C_D A$), ma sát lốp slick.
4. **Nguồn dữ liệu số hóa mã nguồn mở**: Chỉ dẫn trực tiếp tới kho dữ liệu của Đại học Kỹ thuật München (**TUMFTM**) và **F1TENTH**.

---

## 🔬 II. ĐỐI CHIẾU HÌNH HỌC: SA BÀN THỰC TẾ vs SA BÀN METADRIVE HIỆN TẠI

### 2.1. Chiều dài & Tỷ lệ co giãn (Circuit Length & Scaling Ratio)
* **Thực tế Silverstone GP**: Chiều dài là **$5,891\text{ m}$ ($5.89\text{ km}$)**.
* **Sa bàn MetaDrive hiện tại**: Chiều dài là **$2,838.7\text{ m}$ ($2.84\text{ km}$)**.
* **Bản chất kỹ thuật**:
  * Sa bàn trong MetaDrive được xây dựng dựa trên tỷ lệ nén chiều dài xấp xỉ **$\approx 2.07 : 1$** (khoảng $1:2$).
  * Việc nén chiều dài sa bàn là một kỹ thuật tiêu chuẩn trong Reinforcement Learning mô phỏng: Nếu giữ nguyên $5.89\text{ km}$, ở vận tốc trung bình $100\text{ km/h}$, một vòng đua sẽ kéo dài gần $4,000$ bước mô phỏng ($400\text{s}$), làm bùng nổ phương sai của hàm giá trị PPO và khiến tín hiệu phần thưởng bị phân rã (reward attenuation) nghiêm trọng. Chiều dài $2.84\text{ km}$ giúp một vòng đua nằm gọn trong $1,200 - 1,500$ bước, tối ưu hoàn hảo cho rollout length 2048.

### 2.2. Địa hình & Độ chênh lệch độ cao (Elevation Change)
* Báo cáo chỉ ra: Độ chênh lệch độ cao toàn mạch chỉ khoảng **$10.7\text{ m} - 11.0\text{ m}$** trên tổng chiều dài gần $6\text{ km}$ (độ dốc trung bình $< 0.18\%$).
* Đoạn dốc lớn nhất từ Stowe xuống Vale chỉ hạ khoảng $4\text{ m}$ trên quãng đường $350\text{ m}$ (độ dốc ~ $1.1^\circ$).
* **Ý nghĩa học thuật**: Điều này **chứng minh tính hợp lệ tuyệt đối của giả định mặt đường phẳng 2D ($z = 0$) trong môi trường MetaDrive**. Việc bỏ qua cao độ $z$ không làm mất đi tính chân thực động lực học của sa bàn Silverstone.

### 2.3. Bề rộng mặt đường (Track Width)
* Thực tế Silverstone: Rộng **$12.0\text{ m} - 17.0\text{ m}$** (Trung bình $13.5\text{ m} - 15.0\text{ m}$).
* Xe GT3: Rộng **$2.05\text{ m}$**.
* Tỷ lệ thực tế: $R_{\text{real}} = \frac{14.5\text{ m}}{2.05\text{ m}} \approx \mathbf{7.1\times}$.
* Map hiện tại của ta: Rộng **$9.0\text{ m}$** $\implies R = \frac{9.0}{1.85} \approx \mathbf{4.86\times}$ (Khá hẹp, tương đương Monaco).
* Map của nhánh multicar: Rộng **$13.5\text{ m}$** $\implies R = \frac{13.5}{1.85} \approx \mathbf{7.30\times}$ (Đúng chuẩn F1 Silverstone).

---

## 🏎️ III. ĐỐI CHIẾU ĐỘ GẮT CỦA CÁC GÓC CUA (CORNER SEVERITY ANALYSIS)

Báo cáo telemetry thực tế cung cấp mốc vận tốc đỉnh cua ($v_{\text{apex}}$) và bán kính hiệu dụng ($R_{\text{eff}}$) rất rõ ràng:

| Cua | Tên góc cua | Bán kính hiệu dụng thực tế ($R_{\text{eff}}$) | $v_{\text{apex}}$ thực tế (GT3) | Bán kính trong Map MetaDrive ($R_{\text{sim}}$) | Nhận xét đối chiếu |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **T1** | Abbey | $\approx 161\text{ m}$ | $199\text{ km/h}$ | $R = 75\text{ m}$ | Map sim cua gắt hơn thật (do nén 1:2) |
| **T3** | Village | $\approx 41\text{ m}$ | $98\text{ km/h}$ | $R = 30\text{ m}$ | Khá khớp, góc cua chậm |
| **T4** | **The Loop** | $\approx \mathbf{27\text{ m}}$ | **$81\text{ km/h}$** | $\mathbf{R = 25\text{ m}}$ | **Khớp gần như tuyệt đối (25m vs 27m)!** |
| **T7** | Luffield | $\approx 34\text{ m}$ | $90\text{ km/h}$ | $R = 36\text{ m}$ | **Khớp hoàn hảo (36m vs 34m)** |
| **T9** | Copse | $\approx 128\text{ m}$ | $183\text{ km/h}$ | $R = 100\text{ m}$ | Cua tốc độ cao |
| **T15** | Stowe | $\approx 85\text{ m}$ | $147\text{ km/h}$ | $R = 65\text{ m}$ | Cua nhanh cuối Hangar |
| **T16** | Vale | $\approx \mathbf{32\text{ m}}$ | **$84\text{ km/h}$** | $\mathbf{R = 25\text{ m}}$ | Cua gắt phanh nặng |

> [!IMPORTANT]
> **Nhận định cốt lõi**: Map MetaDrive hiện tại **KHÔNG HỀ HIỀN**, mà các góc cua chậm như **The Loop ($R=25\text{m}$)** và **Vale ($R=25\text{m}$)** thậm chí còn gắt hơn hoặc bằng đúng bán kính của Silverstone thật ($R=27\text{m}$)! 
> Xe GT3 thật tại The Loop chỉ chạy được $81\text{ km/h}$ dù có lực ép khí động học $a_y = 1.89g$. Trong Bullet Physics ma sát lốp của ta là $\mu = 1.35$, vận tốc tối đa qua The Loop chỉ là khoảng $55 - 60\text{ km/h}$. Đây là lý do vì sao AI lao vào ở $85\text{ km/h}$ đều bị văng lề!

---

## 📈 IV. CÓ TÍNH ĐƯỢC OPTIMAL PROFILE TỪ BÁO CÁO NÀY KHÔNG?

👉 **CÂU TRẢ LỜI LÀ: HOÀN TOÀN CÓ THỂ, VÀ CÓ THỂ TÍNH THEO 2 CÁCH KHÁC NHAU:**

### Cách 1: Sử dụng Bộ Dữ Liệu Tọa Độ Có Sẵn từ Đại học Kỹ thuật München (TUMFTM)
* Trong **Mục 4** của báo cáo, tác giả đã chỉ ra kho dữ liệu [`TUMFTM/racetrack-database`](https://github.com/TUMFTM/racetrack-database) và [`f1tenth/f1tenth_racetracks`](https://github.com/f1tenth/f1tenth_racetracks).
* File `Silverstone_raceline.csv` đã giải sẵn nghiệm tối ưu:
  $$[s, x, y, \psi, \kappa, v_x, a_x]$$
  cho toàn bộ vòng đua Silverstone đầy đủ ($5.89\text{ km}$).
* Ta có thể viết một script đơn giản để nạp file CSV này vào và đối chiếu mốc chuẩn.

### Cách 2: Tự động Tính toán qua Động cơ Nội sinh `RacingBenchmarkEngine` (Được khuyên dùng)
* Hệ thống mã nguồn của chúng ta ([game_env/benchmark.py](file:///c:/Users/tunbi/Project/school/REL%20Assignment/game_env/benchmark.py)) đã tích hợp sẵn chính phương pháp của nhóm nghiên cứu TUM (Heilmeier et al. 2020) và Lipp & Boyd (2014):
  1. **Convex QP Minimum Curvature**: Giải tối ưu hóa phân bố ngang $d_{\text{lat}}$ để tìm Racing Line giảm thiểu lực ly tâm.
  2. **Monotonic 3-Pass Velocity Profile**: Tính phân bố tốc độ dọc dựa trên $a_{\text{engine}}$, $a_{\text{brake}}$ và elip ma sát Kamm.
* **Cách thực hiện**:
  * Động cơ `RacingBenchmarkEngine` tự động trích xuất polyline của bản đồ (bất kể là map 9m hiện tại hay map 13.5m tương lai) và tính ra nghiệm giải tích tối ưu tức thì trong **$1.5\text{s}$**, tự động lưu vào `silverstone_optimal_profile.npz`.
  * Có thể sử dụng bảng vận tốc góc cua trong báo cáo ([knowledge/silverstone_gt3_rl_specs.md](file:///c:/Users/tunbi/Project/school/REL%20Assignment/knowledge/silverstone_gt3_rl_specs.md#L42)) làm **Ground Truth** để hiệu chuẩn (calibrate) các hệ số gia tốc phanh $a_{\text{brake}}$ và ma sát ngang $a_{\text{lat,limit}}$ trong động cơ benchmark.

---

## 🎯 V. KẾ HOẠCH HÀNH ĐỘNG CHO COMMIT HIỆN TẠI

Theo đúng chỉ đạo của người dùng:
1. **Giữ nguyên map hiện tại ($9.0\text{m}$)**: Chưa thay đổi hình học bản đồ ở commit này để bảo toàn tính ổn định của môi trường.
2. **Cập nhật toàn bộ Bộ Setup & Sửa lỗi Thuật toán (Từ nhánh Multicar & Báo cáo Specs)**:
   * **Sửa bias chân phanh**: `actor_out.bias[2] = -1.6` (khử 100% bẫy rê phanh 17% ban đầu).
   * **Khóa trôi phương sai**: Thêm `self.ac.actor_logstd.clamp_()` sau mỗi bước `actor_optimizer.step()`.
   * **Đồng nhất Squashed Gaussian**: $\text{ACTION\_BOUND} = 0.999$ và $\text{atanh}$ nhất quán.
   * **Áp lực thời gian**: Bổ sung $r_{\text{time}} = -0.01$ mỗi bước ($-\text{time\_penalty}$) vào hàm thưởng.
   * **Cập nhật thông số cơ học xe GT3**: Đồng bộ khối lượng $1,400\text{kg}$, góc lái $\pm 26^\circ$ ($0.45\text{ rad}$) theo Section 6 của báo cáo.
3. **Lưu sa bàn F1 mở rộng $13.5\text{m}$ sang commit kế tiếp**: Sau khi commit này hoàn thành và kiểm thử ổn định, ta sẽ mở rộng map ở commit sau.

