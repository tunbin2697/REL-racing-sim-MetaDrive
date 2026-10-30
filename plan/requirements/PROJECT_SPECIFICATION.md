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
6. [Outcome & Mốc Benchmark Đối Chiếu](#-6-outcome--mốc-benchmark-đối-chiếu-cập-nhật-31102026)
7. [Các Hướng Mở Rộng](#-7-các-hướng-mở-rộng-cập-nhật-31102026)

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

### 5.4. Trạng thái thực hiện (cập nhật 31/10/2026, nhánh `feature/111s-honest-anchor`)

Kết quả chính là **mô hình tham chiếu `ref_111s`** (`models/ref_111s/best_agent.pt`, commit gốc `ee99f9d`). Các điểm lệch so với §2 và §4 được liệt kê đầy đủ, kèm lý do, ở [SPECIFICATION_INTENDED_VIOLATION.md](SPECIFICATION_INTENDED_VIOLATION.md).

1. **Map Silverstone khép kín** (map half, khoảng 2.85 km, 2 làn × 4.5 m = 9 m): dựng từ các block MetaDrive bằng `scripts/solve_silverstone_layout.py`, vạch đích trên Hamilton Straight. Hết lỗi kết thúc ở 98% (§5.1 mục 5).
2. **Obs 35D, chỉ dùng hình học tim đường**: 10 chiều động học xe (vận tốc, v_x, v_y, yaw rate, lái/ga/phanh bước trước, cos/sin lệch hướng, lệch tim đường); 5 chiều biên đường và tiến trình (khoảng cách tới mép trái/phải, cờ lấn lề, s/L, độ cong); 20 chiều lookahead (5 điểm tim đường ở 10/20/35/50/70 m). Không có dữ liệu tối ưu.
3. **Action 3D** `[lái, ga, phanh]` qua bộ trộn pedal và giới hạn tốc độ xoay vô lăng 0.18/bước.
4. **Reward của `ref_111s`**: thưởng tiến độ dọc tim đường 0.05·Δs; phạt Δlái² (−0.10); **phạt lệch tim đường `r_center = −0.04·(lat/4.5)²`**; phạt đứng yên dưới 10 km/h; kết thúc và phạt theo tốc độ khi ra khỏi đường; thưởng về đích 100 + bậc thang theo lap time. `r_center` lệch với khuyến nghị ở §4 (xem file violation).
5. **Luật 4 bánh** (§2.2): env hiện tại kết thúc khi cả 4 bánh rời mặt nhựa, không có kerb ảo. `ref_111s` được train dưới luật cũ (4 góc thân xe, kerb ảo) và được **eval lại dưới luật 4 bánh**.

---

## 📊 6. Outcome & Mốc Benchmark Đối Chiếu (cập nhật 31/10/2026)

### 6.1. Mốc so sánh độc lập
- `RacingBenchmarkEngine` (`game_env/benchmark.py`): racing line cực tiểu độ cong (QP, biên ngang ±4.05 m) và profile tốc độ tối ưu thời gian (Lipp & Boyd 3 pass, elip ma sát Kamm). Cho ra T\* của mô hình chất điểm.
- `OptimalRacingAgent` (`agents/optimal_agent.py`): ghost bám profile bằng MPC dọc và pure pursuit.
- **Ghost hiệu chỉnh (xe `gt3_anchor`)**: giới hạn gia tốc lấy từ số đo trên Bullet (bám ngang, phanh, tăng tốc), chọn cấu hình nhanh nhất mà ghost vẫn chạy sạch trong 33 cấu hình đã quét. Ghost legacy dùng hệ số chỉnh tay (phanh 6.2 m/s² trong khi xe phanh được khoảng 18 m/s²).
- Profile tối ưu **chỉ** dùng để đối chiếu và chạy ghost; không vào obs hay reward của `ref_111s`.

### 6.2. Kết quả (cùng env, luật 4 bánh, cùng vật lý xe)
| | Vòng xuất phát đứng | Vòng flying | T\* | Sạch |
| :-- | :-- | :-- | :-- | :-- |
| Ghost legacy | 119.5s | 114.2s | 112.59s | 2/2 |
| Ghost hiệu chỉnh | 118.2s | 112.8s | 110.62s | 3/3 |
| **`ref_111s`** (20 episode × 2 vòng) | **114.8 ± 0.4s** | **109.4 ± 0.5s** | – | 18/20 |

Diễn giải: policy RL nhanh hơn bộ điều khiển dựa trên mô hình đã hiệu chỉnh khoảng 3%, và vòng flying nhanh hơn cả T\* của mọi envelope mà ghost chạy được. T\* là cận của **mô hình chất điểm**, không phải cận của simulator (`knowledge/vehicle_and_engine_physics.md` §7).

### 6.3. Các chỉ số đánh giá (`training/evaluator.py`)
Lap time trên vòng sạch (trung bình, độ lệch chuẩn, min), tỉ lệ vòng sạch, vị trí crash; Racing Line MAE/RMSE và Apex Accuracy so với racing line tham chiếu; Corner ΔV; mức dùng elip ma sát; Pace Efficiency so với T\*.

---

## 🌟 7. Các Hướng Mở Rộng (cập nhật 31/10/2026)

1. **Ghost car và demo người với AI**: đã có (`demo/spectate_ai.py --ghost`, `demo/play_vs_ai.py`). Ghost xuyên thấu, không va chạm với xe chính.
2. **Silverstone tỉ lệ 1:1 với xe GT3 gần thật** (Phase 2): đã thử với nhiều cấu hình (obs racing line, phạt ra ngoài đường kiểu GT Sophy, học từ lỗi); best sạch 143.4s so với ghost 124.9s. **Đóng băng**, giữ làm chương "mở rộng đã thử" (`knowledge/ablation_inventory.md` R12–R14b).
3. **Đua nhiều xe**: khung 6 xe self-play đã có trên `origin/feature/multicar-selfplay`; bản demo cuối do team làm theo `plan/team_handoff_multicar.md`.

---
*Tài liệu này là đặc tả yêu cầu gốc của đồ án. Mục 1–5.3 giữ nguyên như lúc giao đề; mục 5.4–7 ghi trạng thái thực hiện.*
