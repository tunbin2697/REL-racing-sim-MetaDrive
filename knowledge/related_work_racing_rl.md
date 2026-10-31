# Nghiên cứu liên quan: RL đua xe (GT Sophy, Fuchs et al.) và so sánh với dự án

> **Tạo**: 30/10/2026. Dùng cho phần "Nghiên cứu liên quan" và "Hạn chế" của báo cáo, và làm cơ sở cho D30–D33 trong `plan/archive/plan_v12_full.md`.
> **Nguồn đã đọc trực tiếp**:
> - Wurman et al., *Outracing champion Gran Turismo drivers with deep reinforcement learning*, Nature 602, 223–228 (2022). [PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf) · [bib](https://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-nature22.html)
> - Fuchs, Song, Kaufmann, Scaramuzza, Dürr, *Super-Human Performance in Gran Turismo Sport Using Deep Reinforcement Learning*, IEEE RA-L / ICRA 2021. [arXiv 2008.07971](https://arxiv.org/html/2008.07971v2)
>
> Mọi con số dưới đây lấy từ hai bài trên, trừ khi ghi là của dự án.

---

## 1. Fuchs et al. 2021: time trial một xe

### Cách train
| Thành phần | Chi tiết |
| :-- | :-- |
| Thuật toán | SAC. Policy, 2 Q-network và value network, mỗi mạng 2×256 ReLU. γ = 0.98 / 0.982, lr 3e-4, batch 4096, replay 4M, TD 5 bước |
| Tần số | 10 Hz khi train (game chạy 60 Hz), 60 Hz khi eval |
| Obs | Vận tốc và gia tốc 3D; góc lệch hướng so với tim đường; 13 tia đo khoảng cách (180° phía trước, tầm 100 m); độ cong tim đường ở 10 điểm từ 1.0 đến 2.8 s phía trước; lệnh lái bước trước; cờ chạm tường. **Không có racing line** |
| Action | Lái + **1 trục ga/phanh gộp** [−1, 1]. Hộp số tự động |
| Reward | Tiến độ dọc tim đường $- c_w\|v\|^2$ khi chạm tường ($c_w = 5\times10^{-4}$). Phạt cố định thì bị bỏ qua hoặc làm agent phanh đứng |
| Kết thúc episode | **Không** kết thúc khi chạm tường |
| Tài nguyên | 4 PS4 × 20 xe + 1 PC (GTX 1080 Ti). 56–73 giờ để vượt người nhanh nhất (3 seed), khoảng 946.000 km lái |
| Ghi chú | PPO tốn dữ liệu hơn nhiều và hội tụ sớm. Behavior cloning chỉ dùng để chọn feature; warm-start từ đó không giúp |

### Benchmark
| Setting | Agent | Người nhanh nhất | Trung vị người | AI có sẵn |
| :-- | :-- | :-- | :-- | :-- |
| A: Audi TT Cup | 1:15.913 | 1:16.062 | 1:22.300 | 1:26.899 |
| B: Mazda Demio | 1:39.408 | 1:39.445 | 1:47.259 | 1:52.075 |
| C: Audi TT, đường có hairpin | 2:06.701 | 2:07.319 | 2:13.980 | 2:14.252 |

- **Độ ổn định**: độ lệch chuẩn lap time qua 10 vòng là 15 ms, so với 480 ms của người top 1%.
- **Độ bền**:
  - Nhiễu obs ≤ 2%: gần như không đổi. Nhiễu 9%: vẫn về đích nhưng chậm hơn.
  - Trễ action 50 ms: mất 0.3 s. Trễ 100 ms: phanh muộn, chạm tường. Trễ 150 ms: không còn cạnh tranh.
  - Đổi xe: vẫn trong đường nhưng vào cua gắt bị quá đà. Đổi đường: kẹt ở hairpin.
- **So line**: định tính, gồm chồng quỹ đạo, speed trace và chênh tốc độ từng đoạn. **Không có metric độ lệch line.**

---

## 2. GT Sophy (Wurman et al., Nature 2022): time trial và đua trực tiếp

### Cách train
| Thành phần | Chi tiết |
| :-- | :-- |
| Thuật toán | **QR-SAC**: SAC với critic phân phối, 32 quantile, TD N bước (5 bước ở bản tháng 7, 7 bước ở bản tháng 10). Mạng 4×2048 ReLU, policy có dropout 0.1. γ = 0.9896, α = 0.01, lr 5e-5 (Q) / 2.5e-5 (policy), replay 10⁷, batch 1024, clip gradient Q ở norm 10 |
| Tần số | 10 Hz (đã thử 5–60 Hz, trên 10 Hz không lợi thêm) |
| Action | 2 chiều: lái + **1 trục ga/phanh** [−1, 1]. Game cho 3 trục nhưng họ gộp vì ga và phanh hiếm khi cùng lúc. Gaussian squashed tanh, **logσ trong (−40, 4)** |
| Obs xe | Vận tốc, vận tốc góc, gia tốc 3D; **tải và góc trượt từng lốp**; tiến độ trên vòng (sin/cos); độ dốc mặt đường; hướng so với tim đường; lái/ga/phanh bước trước; cờ chạm tường, cờ ra ngoài đường (≥ 3 bánh ngoài biên, theo game) |
| Obs đường | 60 điểm 3D trên mỗi đường: mép trái, tim đường, mép phải. **Phủ ~6 s phía trước, co giãn theo tốc độ.** **Không có racing line.** Ablation: biểu diễn bằng điểm tốt hơn kiểu tia đo + độ cong |
| Obs đối thủ | Danh sách xe trước và sau trong một khoảng cố định: vị trí, vận tốc, gia tốc tương đối; cờ va chạm; chỉ số slipstream |
| Chuẩn hóa | z-score theo khoảng giá trị đã biết của từng feature (giả định phân phối đều) |
| Reward | Tổ hợp tuyến tính chỉnh tay theo từng đường đua (xem bảng dưới) |
| Kết thúc episode | **Không** kết thúc khi ra ngoài đường. Kịch bản chạy cố định 150 s |
| Kịch bản | 1v0, 1v1, 1v2, 1v3, 1v7; random vị trí, tốc độ đầu, khoảng cách, policy đối thủ. Thêm kịch bản chuyên biệt (xuất phát 8 xe, vượt bằng slipstream, chicane cuối). **Mistake learning**: khi eval mà mất lái, lưu trạng thái ngay trước đó làm điểm xuất phát mới. Lấy mẫu phân tầng theo kịch bản để không quên kỹ năng |
| Tài nguyên | Time trial: **10 PS4** × 20 xe, khoảng 8 ngày, trên 45.000 giờ lái. Đua: 21 PS4, 7–12 ngày. Trainer 1 V100 hoặc nửa A100. **15 seed đều đạt mức siêu người**; phần lớn sau ~10 ngày, lâu nhất 25 ngày |

**Các thành phần reward**
| Thành phần | Định nghĩa |
| :-- | :-- |
| Tiến độ $R_{cp}$ | Chênh lệch quãng đường dọc tim đường. **Bị tắt khi đang ở ngoài đường**, để không có lợi khi cắt cua |
| Ra ngoài đường | −(thời gian ngoài đường tăng thêm) × v² (km/h). Riêng Sarthe dùng × v, và nhân đôi ở 2 chicane |
| Chạm tường | −(thời gian chạm tường tăng thêm) × v² |
| Trượt lốp | Phạt khi lốp trượt khác hướng bánh (không dùng ở Sarthe) |
| Vượt xe | ± tiến độ tương đối so với từng đối thủ trong cửa sổ 20 m phía sau đến 40 m phía trước. Đối xứng, nên không có vòng lặp tự cộng điểm |
| Va chạm nói chung, húc đuôi (∝ v²), va chạm phi thể thao | Dạy luật đua. Tăng phạt va chạm quá mức làm agent rụt rè |

### Benchmark
- **Time trial**: lap time so với phân phối best lap của trên 17.700 người (Maggiore) và trên 12.000 người (Seaside). Histogram 100 vòng: **lap trung bình của Sophy ≈ best lap của người giỏi nhất**. Trong trận time trial trực tiếp (7/2021), Sophy thắng cả 3 đường, dù người được xem ghost của Sophy.
- **Đua 4v4** (tính điểm đội 10-8-6-5-4-3-2-1, Sarthe nhân đôi): thua **86–70** ngày 2/7/2021; thắng **104–52** ngày 21/10/2021 sau khi tăng kích thước mạng, chỉnh feature/reward và cải thiện pool đối thủ.
- **Ablation** (mỗi cấu hình 5 seed): SAC vs QR-SAC (không có QR thì không nhanh hơn người); cách biểu diễn đường; bỏ phạt ra ngoài đường (chậm hơn và ít vòng giữ trong biên hơn); N bước 1→9; loại đối thủ (chỉ AI có sẵn thì quá hung hăng, đối thủ hung hăng thì mất sức cạnh tranh); các mức phạt va chạm.
- **Metric phụ**: "questionable collisions", tỉ lệ vòng không ra ngoài đường, tỉ lệ thành công bài test slipstream.
- **Không có metric độ lệch racing line.**
- **Hạn chế họ tự nêu**: chiến lược còn yếu, ví dụ vượt sớm trên đoạn thẳng nên bị vượt lại bằng slipstream.

---

## 3. So sánh với dự án

> Bảng dưới so với **Phase 2** (map 1:1), đã đóng băng. Kết quả chính của đồ án là `ref_111s` trên map half: PPO, pedal 2 trục, kết thúc khi ra ngoài đường, 4M step (~111 giờ lái), 2 seed. Xem dòng "Dự án hơn ở đâu".

| | GT Sophy | Fuchs 2021 | Dự án (P2-single) |
| :-- | :-- | :-- | :-- |
| Thuật toán | QR-SAC | SAC | PPO |
| Racing line trong agent | Không | Không | Không (nhánh `pure`); có ở nhánh `line` (D30) |
| Obs đường | Điểm mép + tim đường, ~6 s, co giãn theo tốc độ | Tia đo + độ cong 1–2.8 s | Tim đường ở 15/40/80/130/200 m cố định (~2.8 s ở 260 km/h) |
| Obs lốp | Tải và góc trượt từng lốp | Không | Không |
| Action pedal | 1 trục | 1 trục | 2 trục, gộp bằng `thr·(1−brk)−brk` với ngưỡng `brk > 0.05` |
| Phạt va chạm/ra ngoài | ∝ v², chạy tiếp | ∝ v², chạy tiếp | $-10-20(v/v_{max})^2$, **kết thúc episode** |
| Điểm xuất phát | Nhiều kịch bản + mistake learning | Nhiều xe song song trên đường | Chỉ grid (D32 thêm mistake learning) |
| Lượng dữ liệu | > 45.000 giờ | ~946.000 km | 6M × 0.1 s ≈ 167 giờ ≈ 20.000 km |
| Số seed | 15 (ablation 5) | 3 | 1 |
| Mốc đối chứng | Phân phối lap time của người | Phân phối lap time của người | Cận giải tích T\* + ghost |
| Metric | Lap time, histogram, điểm đua, va chạm | Lap time, độ lệch chuẩn, độ bền, so line định tính | Lap time, completion, crash rate, Line MAE, Apex, ΔV cua, Grip, Pace Eff |

### Dự án hơn ở đâu
- Có mốc đối chứng vật lý (T\*, ghost) và metric định lượng về line, cua, độ bám. Hai bài kia không có, vì mốc của họ là người.
- `ref_111s` (map half), eval dưới luật 4 bánh: 114.8 / 109.4s (đứng / flying) so với ghost đã hiệu chỉnh 118.2 / 112.8s; seed 43 cũng nhanh hơn ghost (F24). Tương tự "vượt AI có sẵn" của họ, nhưng mốc là bộ điều khiển dựa trên mô hình giải tích, đã được làm mạnh nhất có thể, chứ không phải người.

### Dự án kém ở đâu, và liên hệ với triệu chứng P2 (F17)
1. **Kết thúc episode khi ra ngoài đường.** Mỗi lỗi mất hết phần còn lại của vòng, nên agent ít gặp nửa sau map và học cách chạy dè dặt. Cả hai bài đều phạt rồi cho chạy tiếp.
2. **Pedal 2 trục.** Cả hai bài gộp thành 1 trục. Đây là ứng viên hợp lý cho kiểu pedal giật đo được ở F17.
3. **Dữ liệu ít hơn hàng trăm lần, lại dùng PPO.** Fuchs ghi rõ PPO tốn dữ liệu và hội tụ sớm.
4. **Không có obs lốp**, và lookahead cố định theo mét, ngắn hơn ở tốc độ cao.
5. **1 seed.** Mọi so sánh ablation chỉ là định tính.

### Áp dụng vào dự án
| Ý tưởng | Áp dụng | Ghi chú |
| :-- | :-- | :-- |
| Mistake learning | D32, WI-20, WI-23 | Không đổi obs nên resume được |
| Histogram lap time, tỉ lệ vòng sạch, độ bền nhiễu obs/trễ action, speed trace | D33, WI-21 | Chỉ cần eval |
| Pedal 1 trục, ra ngoài đường thì phạt ∝ v² và chạy tiếp, mistake learning từ đầu run | **D35**, P2-single v2 và P2-line (`--action-mode combined --offtrack-mode penalty --spawn-mode mistake`) | Train lại từ đầu; v1 (setup cũ) giữ làm mốc so sánh |
| SAC/QR-SAC, obs lốp, lookahead theo thời gian, episode cố định thời gian | Hướng phát triển (`plan/archive/plan_v12_full.md` mục D) | Cần viết lại vòng lặp train hoặc đổi obs |

---

## 4. Cách trình bày nhánh PPO + racing line trong báo cáo
- Không gọi là "data leakage". Trong ML, data leakage là khi thông tin của tập đánh giá lọt vào lúc train. Ở đây là **thông tin đặc quyền / prior từ tối ưu hóa**, đưa vào có chủ ý và khai báo rõ.
- Không viết "đội F1 làm như vậy". Đội F1 tối ưu lap time bằng lap-time simulation (tối ưu điều khiển) và simulator có người lái. Các giải xe tự lái thật chủ yếu tính racing line offline rồi dùng MPC bám theo. Hai bài RL nổi tiếng nhất ở trên đều **không** dùng racing line. Nếu muốn dẫn nhánh "RL + racing line" thì đó là một hướng nghiên cứu riêng (ví dụ trajectory-aided learning cho F1TENTH, cần kiểm tra trích dẫn trước khi dùng).
- Line MAE và Apex Accuracy của nhánh line **không độc lập**, vì agent được nhìn chính racing line đó. So 3 nhánh bằng lap time, completion, crash rate, tỉ lệ vòng sạch.

---

## 5. Paper bổ sung: điều khiển mượt, bang-bang, racing line trong reward (31/10/2026)

### 5.1 Hai bài GT Sophy, Fuchs có lái mượt không, và họ chứng minh điều gì
- **GT Sophy có bang-bang/giật và tự thừa nhận** [đọc trực tiếp PDF]: mục "Fifth, action frequency" ghi Sophy điều khiển 10 Hz, *"much less frequent than human actions"*; Extended Data Fig. 2b–c đặt tín hiệu lái và ga/phanh của Sophy cạnh Igor Fraga ở cua 20 Sarthe, mô tả tay đua có *"much smoother actions"*, *"smooth, 60-Hz signals"*. Họ không sửa, vì không phải mục tiêu.
- **Thứ hai bài chứng minh là lap time siêu người**, mốc so là phân phối lap time của người (và với Sophy, thắng đua trực tiếp có chơi đẹp). **Không bài nào so với nghiệm tối ưu toán học**, và không bài nào có chỉ số độ mượt hay "lái đẹp".
- Thành phần của GT Sophy liên quan tới đua gọn: **phạt trượt lốp** (lốp trượt khác hướng bánh) và phạt ra ngoài đường ∝ v². Dự án chưa dùng phạt trượt lốp; đây là ứng viên nếu demo còn drift.
- Hệ quả cho dự án: mục tiêu "đua đẹp, giống tay đua" của P2 (D37) là mục tiêu **riêng**, cần chỉ số đo được (F20/F23: số lần đổi pedal, đảo lái, β p95) và có cơ sở từ các bài dưới.

### 5.2 Seyde et al., NeurIPS 2021: "Is Bang-Bang Control All You Need? Solving Continuous Control with Bernoulli Policies"
- [arXiv 2111.02552](https://arxiv.org/abs/2111.02552) · [NeurIPS](https://papers.neurips.cc/paper_files/paper/2021/file/e46be61f0050f9cc3a98d5d2192cb0eb-Paper.pdf). Mới kiểm qua abstract và trang hội nghị, chưa đọc toàn văn.
- Thay Gaussian bằng Bernoulli chỉ chọn hai cực mỗi chiều action (bang-bang) mà vẫn đạt kết quả hàng đầu trên nhiều benchmark điều khiển liên tục. Họ liên hệ với bài toán thời gian ngắn nhất, nơi bang-bang thường là nghiệm tối ưu, và lưu ý rằng trên phần cứng thật, chi phí năng lượng và bảo trì khiến bang-bang không chấp nhận được.
- Dùng trong báo cáo: bang-bang là hiện tượng chung của RL điều khiển liên tục, không phải lỗi riêng của dự án; và nó có cơ sở tối ưu (Pontryagin) ở quy mô vài giây, khác với chattering (F23).

### 5.3 Mysore, Mabsout, Mancuso, Saenko, ICRA 2021: "Regularizing Action Policies for Smooth Control with Reinforcement Learning" (CAPS)
- [arXiv 2012.06644](https://arxiv.org/abs/2012.06644) · [trang dự án](https://ai.bu.edu/caps/index.html) · DOI 10.1109/ICRA48506.2021.9561138. Mới kiểm qua abstract và trang dự án.
- Thêm hai số hạng điều chuẩn: **mượt theo thời gian** (action liên tiếp gần nhau) và **mượt theo không gian** (state gần nhau cho action gần nhau). Trên quadrotor thật: tín hiệu mượt hơn nhiều và tiết kiệm năng lượng đáng kể (số % cụ thể cần đọc lại bài trước khi trích).
- Dùng trong báo cáo: cơ sở cho phạt Δlái² và Δpedal² (D38), với lý do giảm hao mòn cơ cấu chấp hành, đúng như người dùng nêu. Phạt của dự án là bản đơn giản của số hạng thời gian, đặt trong reward thay vì trong loss.

### 5.4 Evans, Engelbrecht, Jordaan, IEEE RA-L 2023: "High-speed Autonomous Racing using Trajectory-aided Deep Reinforcement Learning" (TAL)
- [arXiv 2306.07003](https://arxiv.org/abs/2306.07003). Mới kiểm qua abstract; **chưa xác nhận công thức reward cụ thể**.
- Đưa racing line tối ưu vào công thức học của DRL (TD3, F1TENTH sim, 4 map), agent chỉ nhận LiDAR. Kết quả: tỉ lệ hoàn thành vòng ở tốc độ cao cao hơn rõ so với baseline; agent học được profile tốc độ khả thi và bám gần racing line.
- Dùng trong báo cáo: chứng minh "RL + racing line" là một hướng nghiên cứu có thật, không chỉ GT Sophy/Fuchs (RL thuần). Liên hệ với nhánh P2-line và thử nghiệm r_racingline (D36, F21).

### 5.5 Thực tế ngoài học thuật (để viết đúng, tránh nói quá)
- Đội F1: tối ưu lap time bằng lap-time simulation (tối ưu điều khiển) và simulator có người lái; không có RL lái xe đua.
- Đua xe tự lái (Indy Autonomous Challenge, A2RL, F1TENTH): thường tính racing line offline rồi bám bằng MPC/pure pursuit.
- AI trong game đua: thường bám spline racing line cộng profile tốc độ viết sẵn.
- Vậy câu đúng là: GT Sophy và Fuchs **cố ý** không dùng racing line vì câu hỏi của họ là RL tự khám phá cách đua; còn dùng racing line là thông lệ trong đua tự lái và có hướng RL riêng (TAL).
