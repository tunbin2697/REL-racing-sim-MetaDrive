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

## 📊 6. Outcome Kỳ Vọng & Mốc Benchmark Tính Tay

Kết quả đầu ra mà tôi kỳ vọng nhóm mình phải đạt được bao gồm:

### 6.1. Xây dựng mốc so sánh độc lập (Reference Benchmark)
* Team mình cần xây dựng một mốc thời gian cơ sở (có thể tính tay ước lượng racing line lý thuyết, tính toán bán tự động hoặc tìm nguồn đối chiếu hợp lý / make sense) để làm thước đo so sánh độc lập.
* **Quy tắc**: Mốc thời gian đối chiếu này hoàn toàn được xác định độc lập bên ngoài, **tuyệt đối không được leak vào quá trình train của agent**.

### 6.2. Outcome mong đợi
1. **Chạy giống racing line tối ưu nhất**: Agent tự học được đường chạy tiệm cận với racing line tối ưu (tính ngoài, không mớm).
2. **Thời gian vòng đua hợp lý & tiệm cận mốc chuẩn**: Vì chưa xác định cụ thể cách tính tay như thế nào, nên trước mắt lap time thực tế của agent sau huấn luyện chỉ cần đạt mức **tiệm cận, hợp lý (make sense)** và **có nguồn đối chiếu rõ ràng**.
3. **Hoàn thành trọn vẹn 1 lap**: Chạy hết vòng đua mà không vi phạm quy tắc 4 bánh ra ngoài lề đường.

---

## 🌟 7. Các Hướng Mở Rộng Khuyến Khích (Bonus)

> [!TIP]
> **ĐIỀU KIỆN TIÊN QUYẾT:**  
> Toàn bộ các định hướng mở rộng dưới đây **chỉ khuyến khích nhóm mình triển khai nếu việc train agent cốt lõi đã đạt kết quả tốt (hoàn thành đầy đủ các outcome ở Mục 6) và chúng ta vẫn còn thời gian trước ngày báo cáo đồ án**.  
> Ưu tiên hàng đầu của team mình vẫn là làm thật tốt bài toán 1 agent tối ưu lap time trước, tránh phân tán nguồn lực khi mục tiêu chính chưa hoàn thiện.

Nếu đã đảm bảo được kết quả chính và còn thời gian, tôi cực kỳ khuyến khích nhóm mình triển khai thêm các tính năng sau để nâng cao chất lượng đề tài:
1. **Huấn luyện đua đối kháng $> 2$ agents**: Cho các agent cùng so tài, tự học cách né tránh va chạm và tranh chấp vị trí đua.
2. **Người chơi đua trực tiếp cùng agent (Human vs AI)**: Xây dựng chế độ cho phép người chơi điều khiển xe qua bàn phím (`W-A-S-D`) để trực tiếp so tài tốc độ với AI thời gian thực.
3. **Agent có thể đua trên nhiều map F1 thực tế**: Mở rộng kiểm thử agent trên các cấu hình đường đua danh tiếng như Monza, Silverstone,... *(Lưu ý: việc tạo map mới cũng bắt buộc phải tuân thủ nghiêm ngặt các quy luật environment đã nêu ở Mục 2: setting mặc định, không rào chắn sát mép, quy tắc 4 bánh)*.

---
*Tài liệu này là căn cứ yêu cầu chính thức giữa các thành viên trong nhóm đồ án môn học REL.*
