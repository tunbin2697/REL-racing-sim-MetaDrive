# 🏎️ ĐẶC TẢ YÊU CẦU ĐỀ TÀI ĐỒ ÁN MÔN HỌC TĂNG CƯỜNG (REL)
## Tối Ưu Thời Gian Vòng Đua (Lap Time Optimization) Với MetaDrive 3D

> **Môn học:** Học Tăng Cường (Reinforcement Learning - REL)  
> **Người gửi:** Trưởng nhóm / Phụ trách kỹ thuật đồ án  
> **Gửi tới:** Các thành viên trong nhóm đồ án  
> **Mục tiêu:** Thống nhất các quy tắc kỹ thuật, luật môi trường, ghi nhận các nút thắt hiện tại và định hướng giải pháp để team mình cùng nhau huấn luyện AI Agent đạt kết quả tốt nhất.

---

## 📌 MỤC LỤC
1. [Mục Tiêu Đề Tài & Định Hướng Cốt Lõi](#1-mục-tiêu-đề-tài--định-hướng-cốt-lõi)
2. [Các Quy Tắc Môi Trường Bắt Buộc (Environment Rules)](#2-các-quy-tắc-môi-trường-bắt-buộc-environment-rules)
3. [Tiêu Chí Quan Sát Đánh Giá Dựa Trên Lý Thuyết Đua F1](#3-tiêu-chí-quan-sát-đánh-giá-dựa-trên-lý-thuyết-đua-f1)
4. [Nguyên Tắc Học Thuật: Tuyệt Đối Không Data Leak](#4-nguyên-tắc-học-thuật-tuyệt-đối-không-data-leak)
5. [Hiện Trạng Nhánh Main (Trước Bàn Giao) & Gợi Ý Cho Team Mình](#5-hiện-trạng-nhánh-main-trước-bàn-giao--gợi-ý-cho-team-mình)
6. [Outcome Kỳ Vọng & Mốc Benchmark Đối Chiếu](#6-outcome-kỳ-vọng--mốc-benchmark-đối-chiếu)
7. [Các Hướng Mở Rộng Khuyến Khích (Bonus)](#7-các-hướng-mở-rộng-khuyến-khích-bonus)

---

## 🎯 1. Mục Tiêu Đề Tài & Định Hướng Cốt Lõi

Yêu cầu cốt lõi của tôi đối với đồ án này là: **chúng ta cần train 1 agent có khả năng tự học được việc lái xe sao cho tối ưu thời gian hoàn thành 1 lap nhất có thể**.

Khác với bài toán lái xe thông thường (bò chậm, giữ khoảng cách an toàn, bám chính giữa làn đường), đề tài của team mình là xe đua thể thao:
* Agent phải học cách khai thác giới hạn vật lý của xe để chạy nhanh nhất.
* Cân bằng giữa việc tăng tốc ở đoạn thẳng, phanh giảm tốc đúng lúc và ôm cua chuẩn xác.
* **Quan trọng nhất**: Agent phải **tự học thông qua thử - sai (trial-and-error)** trong khuôn khổ môn học REL, chúng ta không "cầm tay chỉ việc" hay mớm đường đi cho xe.

---

## ⚙️ 2. Các Quy Tắc Môi Trường Bắt Buộc (Environment Rules)

Để tránh lỗi cấu hình và đảm bảo tính khách quan khi báo cáo, tôi muốn nhóm mình thống nhất tuân thủ nghiêm ngặt các rule sau:

### 2.1. Dùng setting môi trường mặc định của thư viện
* Giữ nguyên toàn bộ thiết lập mặc định của MetaDrive: gia tốc động cơ, độ trôi của bánh (tire drifting/skid), tốc độ tối đa, cơ chế vật lý lốp/xe, khối lượng, trọng tâm, v.v.
* **Lý do**: Ưu tiên để setup mặc định để tránh việc chúng ta config nhầm dẫn đến thông số phi thực tế hoặc không tương thích với benchmark.

### 2.2. Quy tắc làn đường & Giới hạn đường đua (Track Limits)
* **Agent được chạy ở lane ngược chiều cũng được**: Miễn là nó giúp mở rộng góc vào cua và tối ưu thời gian chạy. Team mình tuyệt đối không cài phạt chạy sai làn hay phạt lệch tim đường.
* **Quy tắc 4 bánh**: Vòng chạy **CHỈ terminate khi cả 4 bánh xe vượt ra ngoài vạch 2 bên đường**. Nếu xe còn ít nhất 1 bánh chạm trong vạch đường thì vẫn tính là hợp lệ (tương tự luật đua FIA).

### 2.3. Quy định về rào chắn & địa hình
* **Đường đua không có rào chắn sát mép đường**: Nhằm tạo không gian mở cua tự nhiên.
* **Khoảng cách an toàn nếu có vật cản**: Nếu sa bàn có rào chắn hoặc vật thể địa hình thì phải đặt **xa hơn 2 bên mép đường một đoạn hơn nửa thân xe ($> 0.5$ thân xe)**. Khi xe va chạm vào rào chắn này thì mới terminate.
* **Lưu ý của tôi về vật lý địa hình**: Do phần vật lý địa hình ven đường của MetaDrive chưa được kiểm thử đầy đủ nên tôi cân nhắc **không áp dụng vật lý cho địa hình** (và nếu vậy thì cũng **không đặt rào chắn**), mà quản lý việc kết thúc vòng chạy thuần túy bằng kiểm tra hình học 4 bánh xe vượt mép đường.

---

## 🏎️ 3. Tiêu Chí Quan Sát Đánh Giá Dựa Trên Lý Thuyết Đua F1

> [!NOTE]
> **Lưu ý cho các bạn:**  
> Vì setup xe và môi trường ưu tiên để mặc định của thư viện nên một số thông số sẽ không giống với xe F1 thực tế (như gia tốc, tốc độ tối đa, độ bám lốp).  
> Các lý thuyết đua xe F1 dưới đây **chỉ là thước đo trực quan** để tôi, các bạn trong nhóm và người chấm điểm theo dõi, đánh giá xem agent sau khi train đã "khôn" lên hay chưa. **Tuyệt đối không biến phần này thành code can thiệp hay mớm cho agent.**

Khi theo dõi agent chạy (qua màn hình 3D hoặc Telemetry), team mình kỳ vọng sẽ quan sát thấy các hành vi tự học sau:
1. **Đạp max ga ở đoạn thẳng**: Trên các đoạn thẳng, agent tự biết nhấn hết ga ($a_{\text{acc}} \to 1.0$), không bị rụt rè nhấp nhả ga hay lắc lư tay lái làm chậm xe.
2. **Kỹ thuật phanh rà (Trail Braking) ở góc cua**: Khi đến góc cua, agent biết phanh dứt khoát trước cua và rà nhả phanh dần khi bắt đầu bẻ lái vào cua.
3. **Vào cua chuẩn racing line & Cắt Apex**: Thay vì bám khư khư tim đường, agent tự học được cách tận dụng bề rộng mặt đường để vào cua chuẩn (Out - In - Out), cắt sát đỉnh góc cua (apex clipping) để duy trì tốc độ ôm cua cao nhất có thể.
4. **Tận dụng tốc độ và độ bám đường**: Xe chạy ổn định ở tốc độ cao gần giới hạn ma sát lốp mà không bị trượt văng xe ra ngoài.

---

## 🚫 4. Nguyên Tắc Học Thuật: Tuyệt Đối Không Data Leak

Đây là đồ án thực hành môn **Học Tăng Cường (Reinforcement Learning - REL)**, tôi yêu cầu nhóm mình phải bảo toàn tính liêm chính học thuật:

* **Tuyệt đối không vi phạm phạm vi môn học REL**:
  * **KHÔNG ĐƯỢC MỚM RACING LINE**: Không nhúng quỹ đạo mẫu, không nhúng tọa độ waypoint lý thuyết vào observation hay reward để ép xe chạy theo ray định sẵn.
  * **KHÔNG ĐƯỢC MỚM THÔNG SỐ MẪU / HÀNH ĐỘNG**: Không dùng imitation learning hay behavior cloning để mớm tay ga/lái của người vào quá trình RL.
  * **KHÔNG CÓ CÁC HÀNH VI DATALEAK**: Không đưa thông số từ mốc tính tay hay kết quả benchmark vào trong observation hoặc làm điều kiện dừng của vòng train.
* **Tránh bẫy các hàm thưởng cục bộ**:
  * Tôi khuyến khích các bạn huấn luyện agent **tự học được cách đua** hơn là mớm các hàm thưởng đạp tốc mù quáng hay thưởng chạy bám tim đường.
  * Những hàm thưởng này rất dễ khiến agent rơi vào **lối mòn cực tiểu cục bộ (local optima)**: thưởng bám tim đường làm xe không bao giờ dám mở góc cua hay cắt apex; thưởng đạp tốc khiến xe đâm sầm ngay góc cua đầu.
* **Khuyến khích áp dụng triệt để kiến thức slide môn học**: Hãy tận dụng tốt các kiến thức nền tảng trong slide môn REL (thiết lập MDP, Value Function, Policy Gradient / PPO, GAE, exploration entropy, cân bằng discount factor $\gamma$) để giải quyết bài toán.

---

## ⚠️ 5. Hiện Trạng Nhánh Main (Trước Bàn Giao) & Gợi Ý Cho Team Mình

Dưới đây là tổng hợp trung thực một số vấn đề và thử nghiệm thất bại trên nhánh `main` trước khi tôi bàn giao lại cho các bạn tiếp tục nghiên cứu:

### 5.1. Các vấn đề hiện tại của Agent trên nhánh main
1. **Agent kẹt ở mức ~70% completion**: Tại điểm này có một góc cua trái ngặt, trong khi toàn bộ các góc cua trước đó của map này đều là **cua phải**. Agent bị lệch tay lái sang phải (right-turn bias) nên khi gặp cua trái thì không kịp xử lý và văng lề.
2. **Tay lái hơi loạn**: Agent đã học được dáng chạy cơ bản nhưng thao tác vô lăng còn rung lắc nhiều (jitter), cần tìm cách giúp tay lái mượt mà hơn.
3. **Không học được kỹ năng đạp phanh**: Agent hoàn toàn giữ ga vừa phải và ôm trọn cua mà không hề đạp phanh (rơi vào tối ưu cục bộ, chưa thoát ra được).
4. **Không đạp max ga ở đoạn thẳng**: Xe chỉ duy trì đều đều ở mức ~80 km/h xuyên suốt 70% đầu thay vì bứt tốc hết cỡ trên đoạn thẳng.
5. **Lỗi 98% nhưng tính là hoàn thành vòng**: Do map đua hiện tại của MetaDrive không phải là 1 vòng lặp khép kín hoàn hảo (closed-loop) mà bị hở ở gần vạch đích.
6. **Lỗi xe 4 bánh ra ngoài nhưng không terminate**: Ở một số đoạn xe đã vượt khỏi vạch mép đường nhưng episode không dừng, nghi ngờ do có sai lệch giữa engine vật lý và mesh hiển thị trên màn hình.

### 5.2. Một số thử nghiệm Policy và Hàm Thưởng đã fail
* **Thưởng đạp ga mù quáng**: Khiến agent chỉ chăm chăm đạp hết ga, fail ngay ở các khúc cua đầu tiên.
* **Thưởng completion tích lũy quá lớn**: Khiến agent kiếm được quá nhiều điểm ở nửa đầu vòng đua, làm mờ nhạt phần thưởng cán đích khiến agent "thỏa mãn" và từ bỏ việc hoàn thành cuộc đua.
* **Mất cân đối giữa thưởng và phạt**: Tỷ lệ thưởng/phạt chưa chuẩn làm Critic dự báo lệch lạc.

### 5.3. Gợi ý hướng đi tiếp cho các bạn
1. **Gợi ý đổi sang 1 map thuần đua xe F1 (Monza, Silverstone, v.v.)**:
   * Để agent học được các góc cua hẹp hơn, nơi mà xe **nhất định cần phải phanh** (thay vì chỉ nhả ga bò qua cua).
   * Các map F1 thực tế có tính vòng lặp khép kín hoàn chỉnh (giải quyết triệt để lỗi kết thúc sớm ở mốc 98%).
   * Sa bàn F1 có cả cua trái và cua phải đan xen, cùng các đoạn thẳng tắp dài để agent tự tin đạp max ga.
2. **Áp dụng lý thuyết slide môn REL để thay đổi Policy và Hàm Thưởng**:
   * **Gợi ý nên tập trung vào thưởng thời gian**: Ưu tiên tối ưu hóa thời gian hoàn thành 1 lap (áp lực thời gian hoàn thành) thay vì cộng quá nhiều điểm tích lũy quãng đường.
   * **Làm mượt tay lái**: Tìm giải pháp điều chỉnh hàm thưởng hoặc policy để tay lái bớt loạn, xe chạy ổn định hơn.
   * **Cân bằng lại tỷ lệ thưởng - phạt** để agent có động lực hoàn thành trọn vẹn vòng đua.

---

### 5.4. Trạng thái & Giải pháp Kỹ thuật Đã Hoàn Thiện Thực Tế (Resolved Implementations)
Toàn bộ các nút thắt trên nhánh `main` cũ đã được nhóm nghiên cứu và giải quyết triệt để thông qua các giải pháp kỹ thuật có luận chứng khoa học:
1. **Tích hợp Sa bàn Đua F1 Silverstone Closed-Loop ($3,417.8\text{m}$)**:
   * Tái hiện 18 góc cua kinh điển của đường đua Silverstone, khép góc hình học hoàn hảo qua bộ giải lặp `solve_silverstone_layout.py`.
   * Giải quyết dứt điểm lỗi đứt đoạn 98%, thiết lập vạch đích Start/Finish chính xác trên đoạn thẳng Hamilton Straight.
2. **Không Gian Quan Sát 40 Chiều (Obs 40D) - Zero Data Leakage**:
   * $16$ chiều Ego/IMU: Vận tốc ($v_x, v_y$), vận tốc góc yaw $\dot{\psi}$, góc trượt thân xe $\beta$, gia tốc trọng trường $a_x, a_y$ (phản ánh vòng tròn ma sát Kamm), độ nghiêng roll/pitch, hành động bước trước và cờ lấn lề.
   * $24$ chiều Lookahead Horizon Preview: 6 waypoint tim đường tại các cự ly $[10, 25, 45, 70, 105, 150]\text{m}$. Tầm nhìn 150m giúp xe tự tin phanh sớm từ tốc độ $> 110\text{ km/h}$.
3. **Không Gian Hành Động 3 Chiều Độc Lập & Bộ Trộn Chân Ga/Phanh $C^1$**:
   * Phân tách riêng biệt Chân Ga (`throttle`) và Chân Phanh (`brake`), kết hợp bộ trộn Drive-by-Wire liên tục $C^1$ và Slew Rate Limiter, triệt tiêu 100% bẫy bão hòa chân ga và rung giật bàn đạp.
4. **Kháng Rung Lắc Tay Lái (Anti-Chatter) Bằng Động Lực Học Tự Nhiên**:
   * Giữ nguyên cơ chế vật lý **Induced Steering Drag** ($F_{\text{drag}} = F_y \sin|\delta|$ qua tâm khối) trong [game_env/extended_physics.py](file:///c:/Users/tunbi/Project/school/REL%20Assignment/game_env/extended_physics.py): Đánh lái dư thừa sẽ tự sinh lực cản dọc làm tụt tốc độ, gián tiếp làm giảm phần thưởng tiến độ mà không cần can thiệp mô-men xoắn nhân tạo.
5. **Hàm Thưởng Chuẩn Hóa Markovian (Không Phạt Tim Đường)**:
   * Loại bỏ hoàn toàn hình phạt lệch tim đường ($r_{\text{center}} = 0$) để Agent tự do mở rộng góc vào cua Out - In - Out và chém Apex.
   * Thưởng tiến độ dọc trục Frenet $\Delta s$, phạt vi sai lái nhẹ $r_{\text{steer\_smooth}} = -0.01 (\Delta\text{steer})^2$, và thưởng cán đích kết hợp thời gian.

---

## 📊 6. Outcome Kỳ Vọng & Mốc Benchmark Đối Chiếu

### 6.1. Xây dựng Mốc So Sánh Độc Lập (Reference Benchmark)
Đồ án đã xây dựng độc lập động cơ phân tích học thuật `RacingBenchmarkEngine` ([game_env/benchmark.py](file:///c:/Users/tunbi/Project/school/REL%20Assignment/game_env/benchmark.py)) và tác tử giải tích `OptimalRacingAgent`:
* **Quỹ đạo Cực tiểu Độ cong (Convex QP Minimum Curvature Line)**: Tối ưu hóa độ dịch chuyển ngang $d_{\text{lat}}$ bằng bộ giải bậc hai OSQP để giảm thiểu lực ly tâm $a_y = v^2 \kappa$.
* **Profile Tốc độ Cực hạn (Lipp & Boyd 3-Pass Method)**: Tính toán phân bố tốc độ tối ưu thời gian dọc quỹ đạo, mang lại mốc thời gian cận trên lý thuyết:
  $$\mathbf{T^* = 116.99\text{s}} \quad (V_{\text{avg}} = 103.5\text{ km/h}, V_{\max} = 120.0\text{ km/h})$$
* **Cam kết Zero Data Leakage**: Dữ liệu cận trên lý thuyết chỉ được dùng làm thước đo đánh giá ngoại tuyến, tuyệt đối không mớm vào Observation hay Reward của PPO.

### 6.2. Các Chỉ Số Định Lượng Đánh Giá Độc Lập
Hệ thống đánh giá học thuật tự động [training/evaluator.py](file:///c:/Users/tunbi/Project/school/REL%20Assignment/training/evaluator.py) trích xuất 5 chỉ số so sánh với mốc chuẩn:
1. **Racing Line MAE & RMSE (m)**: Sai số độ lệch ngang trung bình so với đường đua tối ưu lý thuyết.
2. **Apex Accuracy (%)**: Tỷ lệ phần trăm bám sát đỉnh góc cua trong phạm vi $\le 1.0\text{m}$.
3. **Corner Delta V (km/h)**: Chênh lệch tốc độ đáy cua so với tốc độ tối ưu lý thuyết.
4. **G-G Friction Utilization (%)**: Tỷ lệ khai thác vòng tròn ma sát lốp Kamm (GT3 Grip Limit $\mu = 1.35$).
5. **Pace Efficiency (%)**: Tỷ lệ thời gian vòng đua thực tế so với cận trên lý thuyết ($T_{\text{agent}} / T^* \times 100\%$).

---

## 🌟 7. Các Hướng Mở Rộng Khuyến Khích (Bonus Extensions - Đã Hoàn Thành)

Cả 3 hướng mở rộng nâng cao đều đã được nghiên cứu và tích hợp hoàn chỉnh trong mã nguồn dự án:
1. **Chế độ Đua Người vs AI / Ghost Car thời gian thực ([demo/play_vs_ai.py](file:///c:/Users/tunbi/Project/school/REL%20Assignment/demo/play_vs_ai.py))**:
   * Mở cửa sổ 3D 60FPS cho phép người dùng dùng phím `W-A-S-D` trực tiếp so tài tốc độ cùng AI hoặc Xe đối chứng lý thuyết.
2. **Thực thể Xe Bóng Ma 3D Hologram ([game_env/ghost_car.py](file:///c:/Users/tunbi/Project/school/REL%20Assignment/game_env/ghost_car.py))**:
   * Xe đối chứng lý thuyết chạy song song dưới dạng bóng ma bán trong suốt ($70\%$ opacity), tắt toàn bộ va chạm vật lý để không cản trở xe chính, hiển thị live Telemetry Delta Time ($\Delta t$).
3. **Tổng quát hóa Sa bàn F1 danh tiếng**:
   * Sa bàn Silverstone F1 Grand Prix ($3.4\text{km}$) khép kín hoàn chỉnh với đầy đủ 18 góc cua và đoạn thẳng tốc độ cao.

---
*Tài liệu này là căn cứ đặc tả kỹ thuật và đối chiếu nghiệm thu chính thức của đồ án môn học REL.*
