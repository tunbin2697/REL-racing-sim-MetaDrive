# BÁO CÁO PHÂN TÍCH TOÀN DIỆN HỆ THỐNG RACING RL

> **Trạng thái (31/10/2026)**: **lịch sử**, phân tích thô từ giai đoạn đầu (agent kẹt 30–60%). Các lỗi PPO nêu ở đây đã được xử lý và ghi lại có kiểm chứng ở `knowledge/rl_debugging_findings.md` (F1, F2, F13). Các file khác trong `raw_analysis/` là diff và trích xuất thô giữa các nhánh, dùng làm bằng chứng gốc.

**Chẩn đoán nguyên nhân agent kẹt ở ngưỡng 30%–60% map và vận tốc không bứt phá qua 60 km/h**

---

## 1. NHÓM LỖI THUẬT TOÁN RL & MẠNG NƠ-RON (PPO & NETWORKS)

### Vấn đề 1.1: Kẹp giá trị Critic sai tỷ lệ (Critic Value Loss Clipping Trap)
* **Hiện tượng:** Hàm tổn thất của Critic đang kẹp độ chênh lệch dự đoán theo hệ số `clip_coef` nhỏ (khoảng 0.2).
* **Bản chất lỗi:** Hệ số 0.2 này vốn chỉ áp dụng cho tỷ số xác suất của Policy (dao động quanh 1.0). Trong khi đó, thang điểm Return thực tế của bài toán cộng dồn lên đến hàng chục, hàng trăm điểm. Việc ép Value Function chỉ được dao động trong biên hẹp $\pm 0.2$ khiến mạng Critic hoàn toàn bị tê liệt, không thể dự đoán đúng giá trị tích lũy $V(s)$. Khi Critic đoán sai, hàm tính Advantage $A_t$ thông qua GAE bị sai lệch hoàn toàn, làm hỏng tín hiệu cập nhật chính sách của Actor.
* **Giải pháp cốt lõi:** Bỏ hoàn toàn cơ chế clip đối với Critic; sử dụng hàm mất mát MSE thuần túy giữa giá trị dự đoán và Return thực tế.

### Vấn đề 1.2: Lệch pha toán học ở biên hành động (Action Boundary Mismatch)
* **Hiện tượng:** Mạng Actor lấy mẫu hành động tự do từ phân phối Gaussian không chặn, sau đó dùng hàm cắt cứng (clamp/clip) ở ngoài môi trường để giới hạn về $[-1.0, 1.0]$, nhưng xác suất $log\_prob$ lại được tính trên giá trị chưa cắt.
* **Bản chất lỗi:** Phá vỡ tính đúng đắn của Importance Sampling. Khi agent muốn đạp hết ga (+1.0) hoặc phanh kịch sàn (-1.0), gradient cập nhật bị méo mó nghiêm trọng vì phân phối xác suất trong bộ nhớ không phản ánh đúng hành động thực tế đã tác động lên xe.
* **Giải pháp cốt lõi:** Đồng bộ hóa không gian hành động. Sử dụng cấu trúc phân phối bị chặn chuẩn (Squashed Gaussian qua hàm Tanh có bù Jacobian log-determinant) hoặc tính toán $log\_prob$ chuẩn xác trên giá trị đã bị chặn.

---

## 2. NHÓM LỖI NHẬN THỨC MÔI TRƯỜNG (OBSERVATION & PERCEPTION)

### Vấn đề 2.1: Agent bị "mù" góc cua tương lai ở tốc độ cao (No Lookahead Horizon)
* **Hiện tượng:** Không gian quan sát hiện tại phụ thuộc chủ yếu vào 240 tia LiDAR quét 3D.
* **Bản chất lỗi:** Khi bạn tắt tường bê tông và rào chắn lề (`show_sidewalk=False`), các tia LiDAR quét vào khoảng không vô định. Mặt đường phẳng và vạch sơn không tạo ra va chạm 3D để LiDAR phát hiện. Agent chỉ biết xe đang lệch làn khi bánh xe đã thực sự chạm mép đường.
  * Ở tốc độ 30–40 km/h: Phản xạ tức thời vẫn kịp đánh lái.
  * Ở tốc độ 60–120 km/h: Quán tính chuyển động tỷ lệ với bình phương vận tốc. Không có thông tin nhìn xa trước 20m–50m (độ cong khúc cua sắp tới), xe bắt buộc phải văng ra ngoài theo quy luật vật lý trước khi kịp nhận ra khúc cua.
* **Giải pháp cốt lõi:** Bổ sung vector "tầm nhìn đường đua" (Lookahead Waypoints/Curvature) vào Observation. Cung cấp tọa độ lệch và góc cua của các mốc tim đường phía trước xe ở các cự ly ngắn, trung, và dài (ví dụ: 10m, 25m, 50m).

---

## 3. NHÓM LỖI HÀM THƯỞNG & LÝ THUYẾT TRÒ CHƠI (REWARD GAMING)

### Vấn đề 3.1: Bẫy cực tiểu cục bộ từ Delta Window (Asymmetric Sliding Window Exploitation)
* **Hiện tượng:** Agent nhận thưởng Delta liên tục mỗi bước (step-by-step) nếu nhanh hơn kỷ lục phân đoạn cũ.
* **Bản chất lỗi:** Nửa đầu vòng đua (0%–40%) gồm các đoạn thẳng hoặc góc cua dễ, agent rất dễ tối ưu và thu về lượng điểm Delta khổng lồ. Trong khi đó, nửa sau (40%–70%) là cụm cua gắt; nếu liều mạng vào cua mà thất bại thì bị trừ điểm phạt rất nặng (-40). 
  * Xét theo kỳ vọng toán học: Thuật toán nhận ra việc chạy thật nuột ở 40% đầu rồi dừng lại/chấp nhận kết thúc đem lại tổng điểm cao hơn và an toàn hơn nhiều so với việc mạo hiểm lao vào khúc cua khó.
* **Giải pháp cốt lõi:** 
  1. Không cộng điểm Delta theo từng step thời gian; chỉ tính và thưởng Delta khi xe chính thức hoàn thành việc vượt qua ranh giới của một phân đoạn đường (Sector-based completion).
  2. Dùng một vòng chạy mẫu cố định (Reference Lap) phủ 100% bản đồ làm mốc chuẩn thay vì chỉ so với chính agent, tránh việc đoạn đường chưa khám phá bị mất tín hiệu Delta.

### Vấn đề 3.2: Thiếu áp lực thời gian và thiếu thưởng cán đích
* **Hiện tượng:** Phạt thời gian từng bước bằng 0 (`r_step_penalty = 0.0`), thưởng hoàn thành vòng đua bị tắt hoàn toàn (`r_finish = 0.0`).
* **Bản chất lỗi:** Agent không có lý do để chạy nhanh vì chạy càng nhanh rủi ro mất lái càng cao mà không được đền bù thỏa đáng. Đồng thời, không có "củ cà rốt" đủ lớn ở vạch đích để khuyến khích agent mạo hiểm vượt qua rào cản tâm lý ở các góc cua cuối chặng.
* **Giải pháp cốt lõi:**
  1. Áp đặt áp lực thời gian liên tục (trừ điểm nhẹ mỗi step trôi qua) để ép agent phải chạy với tốc độ cao nhất có thể.
  2. Tái lập phần thưởng hoàn thành vòng đua với trọng số lớn, kết hợp điểm thưởng tỷ lệ nghịch với tổng thời gian hoàn thành (Lap Time).

---

## 4. NHÓM LỖI ĐO LƯỜNG VẬT LÝ & HÌNH HỌC ĐƯỜNG ĐUA (GEOMETRY & TRACK LIMITS)

### Vấn đề 4.1: Tính khoảng cách theo quãng đường lết bánh thay vì tiến trình đường đua
* **Hiện tượng:** `cumulative_distance` trong Telemetry được cộng dồn bằng độ dài vector dịch chuyển vị trí thực tế giữa 2 bước.
* **Bản chất lỗi:** Khi xe trượt bánh, xoay vòng (spin), đánh võng hoặc văng ra bãi cỏ, chỉ số này vẫn tăng đều đặn. Điều này làm trượt khóa phân đoạn (`grid_key`), khiến hệ thống so sánh thành tích của agent tại một khúc cua với kỷ lục của một đoạn đường thẳng hoàn toàn khác.
* **Giải pháp cốt lõi:** Chuyển toàn bộ hệ quy chiếu quãng đường sang tọa độ cung dọc theo tim đường đua (Frenet coordinate $s$). Chỉ ghi nhận tiến trình khi xe di chuyển tịnh tiến đúng hướng dọc theo lòng đường.

### Vấn đề 4.2: Cơ chế dừng xe (Termination) vi phạm nguyên lý cắt cua thể thao (Apex/Kerb Cutting)
* **Hiện tượng:** Xe bị dừng cuộc đua và phạt nặng khi mép thân xe cách mép đường quá 1.5m.
* **Bản chất lỗi:** Trong kỹ thuật đua xe thể thao, xe luôn chém qua gờ giảm tốc (kerb) hoặc đưa 1–2 bánh ra lề cỏ để mở rộng góc vào cua (Apex). Ngưỡng kiểm tra cũ tính từ tâm xe hoặc mép ngoài đã kích hoạt án phạt đâm xe ngay khi xe chỉ mới cắt lề để lấy góc, triệt tiêu khả năng tìm ra quỹ đạo đua tối ưu (Racing Line).
* **Giải pháp cốt lõi:** Áp dụng chuẩn luật 4 bánh (FIA Track Limits). Chỉ dừng episode khi toàn bộ 4 điểm tiếp đất của bánh xe đã hoàn toàn nằm ngoài mép đường nhựa và gờ kerb. Khi có 1 đến 3 bánh trên lề, chỉ giảm độ bám/vận tốc do ma sát lề, tuyệt đối không kết thúc sớm episode.

---

## 5. NHÓM LỖI TỪ PHA TIỀN HUẤN LUYỆN (DAGGER / WARMUP MISMATCH)

### Vấn đề 5.1: Xung đột phong cách lái giữa Expert chạy phố và Racing Policy
* **Hiện tượng:** DAgger sử dụng bot mẫu mặc định của MetaDrive (`numpy_expert`) để học trước hành vi.
* **Bản chất lỗi:** Con bot mẫu này là thuật toán lái xe đô thị an toàn (tốc độ thấp ~30–40 km/h, không phanh trễ, không chém lề, chỉ ôm tim làn). DAgger đã đóng băng thói quen lái xe an toàn tốc độ thấp vào trọng số mạng Actor. Khi chuyển sang PPO để học đua tốc độ cao (60–120 km/h), agent mang phản xạ lái của xe chạy phố vào khúc cua gấp với vận tốc cao, dẫn đến mất lái tức thì và bị sốc điểm phạt.
* **Giải pháp cốt lõi:** 
  1. Nếu dùng DAgger/Imitation làm bàn đạp, dữ liệu mẫu phải đến từ thuật toán bám quỹ đạo đua thể thao (như Pure Pursuit tốc độ cao có tính toán giảm tốc theo độ cong cua) hoặc dữ liệu lái tay của người biết đua xe.
  2. Khi chuyển giao sang PPO, phải mở rộng biên độ khám phá của mạng (tăng entropy hoặc độ lệch chuẩn khám phá của hành động) để rũ bỏ định kiến tốc độ thấp.

---

## 6. LỘ TRÌNH TRIỂN KHAI THEO THỨ TỰ ƯU TIÊN

1. **Khắc phục tầng Nhận thức (Observation):** Đưa thông tin độ cong đường phía trước (Lookahead) vào observation để agent "nhìn thấy" góc cua trước khi lao tới ở tốc độ cao.
2. **Sửa lỗi tính toán cốt lõi:** Xóa bỏ Value Clipping trong Critic của PPO; đồng bộ hóa tọa độ phân đoạn Telemetry theo tim đường $s$.
3. **Nới lỏng Track Limits:** Cập nhật điều kiện dừng vòng đua theo chuẩn 4 bánh vượt ngoài mép kerb để xe dám cắt góc cua.
4. **Cấu trúc lại Hàm thưởng:** Bật áp lực thời gian step penalty; biến Delta thành phần thưởng chốt chặn theo Sector; cấp thưởng lớn khi cán đích trọn vẹn vòng đua.
5. **Kích hoạt Reverse Curriculum:** Sử dụng cơ chế hồi sinh trước điểm va chạm (`milestone_respawn`) để agent tập trung giải quyết bài toán kỹ thuật của riêng góc cua 40%–70% thay vì chạy lại vô ích từ vạch xuất phát.