# Nhật ký chẩn đoán & phát hiện kỹ thuật (phục vụ báo cáo và ablation)

> **Tạo**: 29/10/2026
> **Phạm vi**: các nhánh sau bản 111s (`ee99f9d`) → `2055469` (42D) → `0180988` (40D) → working tree hiện tại (35D), và nhánh `origin/feature/multicar-selfplay` (`7ab74a5`).
> **Quy ước nhãn**:
> - **[Đã kiểm chứng]**: có bằng chứng trực tiếp từ checkpoint, log hoặc code.
> - **[Suy luận]**: tính toán từ code và số liệu, chưa đo trực tiếp.
> - **[Giả thuyết]**: cần thí nghiệm để xác nhận.

---

## F1. σ khám phá bị đóng băng ở biên clamp (actor_logstd freeze)

### Hiện tượng [Đã kiểm chứng]
Giá trị `actor_logstd` đọc trực tiếp từ checkpoint, theo thứ tự [steer, throttle, brake]:

| Checkpoint | logstd | Trạng thái |
| :--- | :--- | :--- |
| 111s (`ee99f9d:models/best_agent.pt`) | [−1.952, −0.881, −0.437] | Trong biên |
| 42D 4M (`2055469:models/last_agent.pt`) | [0.002, 0.000, 0.002] | Cả 3 chiều kẹt trần |
| Hiện tại `agent_step_500k.pt` | [−1.461, 0.001, −0.199] | Throttle kẹt trần |
| Hiện tại `agent_step_3000k.pt`, `best`, `last` | [−2.301, 0.001, 0.001] | Steer kẹt sàn; ga, phanh kẹt trần |
| Multicar (`7ab74a5:models/best_agent.pt`) | [−2.300, −1.871, −1.687] | Trong biên (đã có projection) |

Entropy trong log khớp chính xác với giá trị lý thuyết. Entropy của Gaussian 3 chiều là $\sum_i (\tfrac12\ln 2\pi e + \log\sigma_i) = 4.257 + \sum_i \log\sigma_i$:
- 42D: $\sum\log\sigma \approx 0$ → **4.26**, hằng số suốt phần cuối log.
- Hiện tại: $-2.3 + 0 + 0$ → **1.96**, hằng số suốt phần cuối log.
- 111s: 1.40–1.49, có dao động, tức không kẹt.

### Cơ chế [Đã kiểm chứng từ code]
1. `networks.py` kẹp logstd bằng `torch.clamp(self.actor_logstd, min=-2.3, max=0.0)` ngay trong forward. Tham số gốc `actor_logstd` thì không bị ràng buộc.
2. Khi tham số gốc vượt ra ngoài [−2.3, 0], đạo hàm của clamp bằng 0. Mọi gradient đều mất, gồm cả policy gradient lẫn entropy bonus.
3. Không còn lực nào kéo tham số trở lại, nên nó **kẹt vĩnh viễn**. Phần vượt quá biên rất nhỏ (0.001 trong checkpoint) đến từ đuôi momentum của Adam sau khi gradient đã về 0.

### Vì sao ga/phanh trôi lên trần [Giả thuyết]
Entropy bonus (`ent_coef` = 0.005–0.01) tạo một lực đẩy logstd đi lên không đổi. Nếu nhiễu ở bàn đạp không bị phạt đủ mạnh trong advantage, lực đẩy này thắng và logstd chạm trần. Chưa rõ vì sao run 111s không gặp hiện tượng này. Có thể do episode 1 vòng cùng thưởng bậc thang tạo tín hiệu ga/phanh mạnh hơn, cũng có thể chỉ là ngẫu nhiên.

### Hệ quả [Suy luận]
- Phanh: $brk = \tfrac12(\tanh u + 1)$, deadband 0.05, nên phanh bật khi $u > \operatorname{atanh}(-0.9) = -1.472$.
- Với $\sigma_{brake} = 1$ và mean ≈ bias ≈ −0.96: $P(u > -1.472) = P(z > -0.51) \approx$ **70%** số bước rollout có phanh ngẫu nhiên.
  - Con số này lấy bias làm đại diện cho mean. Mean thật thay đổi theo state.
- Policy học trên dữ liệu mà ga/phanh chủ yếu là nhiễu, nên không học được điểm phanh. Điều này nhất quán với crash_rate ≈ 1.0 và completion ≈ 20% trong `logs/training_metrics.json`.
- Eval dùng mean (deterministic), nên σ không trực tiếp làm hỏng eval. Nhưng bản thân mean đã được học dưới nhiễu.

### Bản vá
Lấy từ nhánh multicar:
```python
self.actor_optimizer.step()
with torch.no_grad():
    self.ac.actor_logstd.clamp_(self.ac.LOGSTD_MIN, self.ac.LOGSTD_MAX)
```
Áp dụng cả trong `load()`. Kiểm chứng: checkpoint của nhánh multicar có logstd nằm trong biên.

### Bài học cho báo cáo
- **Entropy tổng che mất lỗi**: cần log σ cho **từng chiều** hành động.
- Không resume từ checkpoint đã kẹt. Projection lúc load chỉ kéo σ về đúng biên, không kéo về giá trị khỏe.

### Cách tái hiện
```python
import torch
c = torch.load("models/last_agent.pt", map_location="cpu", weights_only=False)
sd = c.get("model_state_dict", c)
print(sd["actor_logstd"])
```

---

## F2. log_prob lệch nhau giữa rollout và update (Squashed Gaussian mismatch)

### Hiện tượng [Đã kiểm chứng từ code `0180988`]
- **Lúc rollout**: `u = clamp(rsample, ±5)`, `action = tanh(u)`. Action có thể đạt 0.99991.
- **Lúc update**: `action = clamp(action, ±0.999)`, `u = atanh(action)`, nên |u| ≤ 3.80.
- Khi |u| > 3.8 (hành động bão hòa), log_prob tính lúc update khác log_prob đã lưu lúc rollout. Ratio PPO vì thế bị lệch một cách hệ thống.

### Mức độ [Suy luận]
- Tác động nhỏ khi mean nằm gần 0.
- Tác động đáng kể khi policy học được kiểu đạp hết ga trên đường thẳng (mean của u lớn), hoặc khi σ kẹt ở trần như F1.

### Bản vá
Lấy từ nhánh multicar: định nghĩa `ACTION_BOUND = 0.999` và **luôn** áp `clamp → atanh` cho cả hai đường, rollout lẫn update.

---

## F3. Bias phanh −0.8 gây rê phanh thường trực

### Hiện tượng [Suy luận từ code]
- Bias −0.8: $\tanh(-0.8) = -0.664$, nên brk = 0.168. Giá trị này lớn hơn deadband 0.05, nên mean policy lúc khởi tạo luôn kéo phanh khoảng 17%.
- Bias −1.6: brk = 0.039, nằm dưới deadband.
- Với σ_init ≈ 0.45 (logstd −0.8): $P(u > -1.472) = P(z > 0.285) \approx 39\%$ số bước vẫn có phanh khi sample. Như vậy vẫn còn khám phá phanh, nhưng mean không còn rê phanh.

---

## F4. Resume bỏ qua `--lr`

**[Đã kiểm chứng từ code]**: `Adam.load_state_dict` khôi phục cả `lr` lưu trong checkpoint, nên giá trị `--lr` truyền qua CLI không có tác dụng.

**Bản vá** (nhánh multicar): gọi `agent.set_lr(args.lr)` sau `agent.load()`.

---

## F5. Evaluator cắt vòng chậm

**[Đã kiểm chứng từ code]**: giới hạn `max_eval_steps_per_lap = 1500` tương đương 150s mỗi vòng. Vòng chậm hơn 150s bị cắt và tính là không hoàn thành, làm đường cong học giai đoạn đầu trông tệ hơn thực tế.

**Bản vá** (nhánh multicar): tăng lên 3000.

---

## F6. Thưởng về đích η³ phẳng ở vùng agent đang chạy

**[Suy luận]**: $R = 50 + 100\eta^3$ với $\eta = \text{clip}((180 - T)/90, 0, 1)$.

| T (s) | Bonus | Độ dốc (điểm/s) |
| :-- | :-- | :-- |
| 160 | 1.1 | 0.05 |
| 146 | 5.4 | 0.16 |
| 120 | 29.6 | 0.99 |
| 111 | 45.2 | 1.31 |

Ở vùng 140–180s, chạy nhanh hơn gần như không được thưởng thêm. Để so sánh:
- Bậc thang của bản 111s: +50 nếu dưới 140s, giảm 10 điểm mỗi 20s.
- Tuyến tính 100·η: độ dốc đều 1.11 điểm/s.

**[Giả thuyết]**: dạng η³ làm chậm giai đoạn đầu của quá trình tối ưu lap time.

---

## F7. Độ lớn của time penalty

**[Suy luận]**:
- Reward tiến độ là $0.05\Delta s = 0.005v$ mỗi bước, tức 0.15/bước ở 30 m/s, tổng khoảng 142/vòng.
- $r_{time} = -0.01$/bước, khoảng 7% reward tiến độ, khoảng −11/vòng.
- Tổng phạt tương lai có chiết khấu tối đa là $0.01/(1-\gamma)$: bằng 1 với γ = 0.99, bằng 2 với γ = 0.995. Con số này nhỏ hơn nhiều so với phạt crash (−10 đến −30), nên **không tạo động cơ tự sát**.
- Hiệu ứng tăng tốc dự kiến nhỏ.

---

## F8. Kết quả một xe trên nhánh multicar

**[Đã kiểm chứng từ log]**: các run `run_v5` đến `run_v8b` (mỗi run ≤ 1M step, đã có bản vá F1 và F2) đạt best lap khoảng **146s**, crash_rate cao.

Nhánh này còn cho thấy một hiện tượng ngược lại: σ cả 3 chiều sập gần sàn. Entropy của v7 là −2.62, tương ứng Σlogσ ≈ −6.88 ≈ 3 × (−2.3). Projection chống được việc kẹt nhưng không chống được việc sập. Như vậy:
- Bản vá F1 và F2 là **cần nhưng chưa đủ**.
- Bản 111s cần 4M step để đạt kết quả.

---

## F9. Các biến thay đổi giữa bản 111s và hiện tại (biến gây nhiễu)

| Hạng mục | 111s (`ee99f9d`) | Hiện tại |
| :--- | :--- | :--- |
| logstd max | 0.5 | 0.0 (commit) / 0.5 (working tree) |
| Episode | 1 vòng, standing | 2 vòng, standing |
| Thưởng về đích | 100 + bậc thang | 50 + 100η³ |
| Phạt stall | 0 | −15 |
| `r_center` | −0.04·(lat/4.5)² | Bỏ |
| Track limit | 4 góc thân xe | 4 bánh xe |
| Obs | 35D | 35D (giữa chừng từng là 42D và 40D) |

---

## F10. Các khẳng định vật lý trong tài liệu specs cần sửa

Đã chuyển sang `knowledge/vehicle_and_engine_physics.md` §8.1 (Froude) và §9 (bảng sửa khẳng định cũ).

---

## F11. Reward hiện tại tương đương "delta time so với ghost 2 m/s"; vì sao không cộng thêm delta time

### Delta time viết dưới dạng reward từng bước [Suy luận]
Gọi $t_{ref}(s)$ là thời điểm ghost tham chiếu đi tới vị trí $s$. Delta là $\delta_t = t_{now} - t_{ref}(s_t)$. Reward từng bước:

$$r_t = -(\delta_t - \delta_{t-1}) = \underbrace{t_{ref}(s_t) - t_{ref}(s_{t-1})}_{\approx\,\Delta s / v_{ref}(s)} - \Delta t$$

Ba hệ quả:
- **Tổng không chiết khấu triệt tiêu theo kiểu telescoping**: $\sum r_t = T_{ref} - T_{lap}$. Tổng này tuyến tính chính xác theo lap time.
- **Số hạng $\Delta s/v_{ref}(s)$ là potential-based shaping** (Ng et al. 1999) với $\Phi(s) = t_{ref}(s)$. Về lý thuyết nó không đổi policy tối ưu, chỉ đổi cách phân bổ credit giữa các bước.
- **Số hạng $-\Delta t$ chính là time penalty.**

### Reward hiện tại là một trường hợp riêng [Suy luận]
So khớp $k(\Delta s/v_g - \Delta t)$ với $0.05\Delta s - 0.01$ (bước 0.1s):
- Từ số hạng thời gian: $k = 0.1$.
- Từ số hạng quãng đường: $v_g = 2$ m/s (7.2 km/h).

Vậy reward tiến độ cộng time penalty 0.01 chính là **delta time so với một ghost chạy 7.2 km/h**. Xe hầu như luôn "nhanh hơn" ghost, nên áp lực thời gian gần như bằng 0. Cũng cần để ý: nếu không có time penalty thì reward tiến độ **cộng dồn không chiết khấu là hằng số $0.05L$**, tức không chứa tín hiệu lap time nào. Tín hiệu đó chỉ xuất hiện nhờ chiết khấu γ.

### Hệ quả nếu cộng chồng các thành phần trùng nhau [Suy luận]
- **Đếm trùng mục tiêu "nhanh"**: tiến độ, time, delta và `r_idle` cùng đẩy về một hướng. Trọng số thực của mục tiêu này bằng tổng các thành phần, lấn át phạt crash và phạt lắc tay lái, nên agent liều hơn và khó tune.
- **Ablation không quy được nguyên nhân** cho riêng thành phần nào.
- **Có thể sinh cực trị cục bộ** ở những state mà các thành phần ngược dấu nhau.
- Trong RL không có vấn đề "không xác định được hệ số" như đa cộng tuyến trong hồi quy, vì agent chỉ tối ưu tổng.

Nguyên tắc rút ra: delta time phải **thay thế** reward tiến độ cộng time penalty, không được cộng thêm vào.

### Delta time chỉ có giá trị khi $v_{ref}(s)$ thay đổi theo vị trí
Khi đó mỗi giây có giá như nhau ở mọi chỗ trên đường, và credit được gán ngay tại khúc cua tạo ra nó. Các lựa chọn nguồn tham chiếu:

| Nguồn tham chiếu | Đánh giá |
| :--- | :--- |
| Optimal profile | **Cấm** (rò rỉ dữ liệu) |
| $\sqrt{a_y/\kappa}$ tính tay | Vùng xám, gần như là Pass 1 của benchmark |
| Best lap của chính agent | Hợp lệ, nhưng non-stationary. Cần tham chiếu dự phòng. Khi agent chạy sát best thì reward trung bình về 0, động lực sống sót biến mất |

**Quyết định**: không triển khai trong Phase 1. Chọn sector bonus hàm mũ thay thế (plan #11). Ghi delta time vào "Hướng phát triển".

---

## F12. Lý do nén map còn 1:2 dựa trên một phép tính sai

**[Suy luận]**: `knowledge/specs_report_analysis.md` viết rằng map 5.89 km chạy ở 100 km/h sẽ cần "gần 4,000 bước (400s)". Tính lại thì: 5,891 m / 27.8 m/s ≈ **212 s, tức khoảng 2,100 bước**. Con số trong tài liệu bị **gấp đôi**.

| Kịch bản trên map 1:1 | Bước / vòng |
| :--- | :--- |
| Nhịp GT3 (lap khoảng 160s, không có khí động học) | khoảng 1,600 |
| Agent ở giai đoạn đầu, khoảng 60 km/h | khoảng 3,500 |
| Map 1:2 hiện tại, lap 111s (để đối chiếu) | khoảng 1,110 |

Như vậy chi phí thật của map 1:1 là khoảng ×1.44 số bước mỗi vòng so với map 1:2, không phải ×3–4. Đây là cơ sở để chọn hướng Y (plan v9).

Hệ quả kèm theo khi chuyển sang 1:1:
- Lookahead phải đủ dài để bao quãng phanh khoảng 180m.
- `v/40` trong obs bị kẹp ở 216 km/h.
- `horizon` và cap step của evaluator phải nhân theo scale.

---

## F13. Ratio PPO nổ tới NaN sau bản vá F2 (hành động bão hòa sâu)

### Hiện tượng [Đã kiểm chứng: run P1-main, commit `e0738e8`]
- Trước khi nổ, run hoạt động tốt:
  - σ = [0.21, 0.91, 0.23], dao động trong biên, tức F1 đã được khắc phục.
  - Vòng sạch đầu tiên 143s; best 141.4s tại step 1,372,160.
- Diễn biến:
  - Step 1,382,400: policy loss = **1.3·10¹⁰**, value loss vẫn ≈ 1.
  - Step 1,392,640: policy loss = 1.2·10⁶; eval crash tại mét 913, completion còn 16%.
  - Update tiếp theo: `actor_mean` ra NaN, chương trình dừng với `ValueError` của `Normal(loc=nan)`.

### Cơ chế [Đã kiểm chứng bằng TEST 4D]
- Bản vá F2 kẹp action về ±0.999, sau đó tính log_prob tại $u = \operatorname{atanh}(\pm0.999) = \pm3.8$ cho **cả rollout lẫn update**. Hai đường đã khớp nhau, nhưng giá trị đó là mật độ tại **điểm bị kẹp**, không phải tại điểm thực sự được sample.
- Agent học cách không phanh trên phần lớn đường đua, nên mean của chiều brake bị đẩy sâu vào vùng bão hòa (khoảng −5). Khi đó điểm −3.8 nằm cách mean khoảng 12σ (σ_brake ≈ 0.1–0.23), tức ở rất xa trong đuôi phân phối.
- Log-density ở đuôi thay đổi rất nhanh theo mean. Chỉ cần mean dịch thêm 1σ về phía biên:
  $$\Delta\log p = \frac{(12\sigma)^2 - (11\sigma)^2}{2\sigma^2} = 11.5 \;\Rightarrow\; r \approx 10^5$$
  TEST 4D đo được: đường cũ cho max ratio = **1.08·10⁵**, đường mới chỉ **10**.
- Clip của PPO không chặn được trường hợp này. Với advantage < 0, số hạng $r\cdot A$ không có cận trên, nên loss nổ, gradient nổ và trọng số thành NaN.
- Vì sao code trước F2 không nổ: log_prob lúc rollout được tính tại $u$ đã sample (gần mean), còn lúc update lại tính tại 3.8 (ở đuôi). Do đó $r \to 0$: gradient bị **triệt tiêu** chứ không bùng nổ. Bản vá F2 làm hai đường khớp nhau nhưng đã chọn sai điểm để khớp.

### Bản vá
- **Lưu $u$ trước tanh vào buffer rollout** (`raw_us`) và tính lại log_prob tại đúng $u$ đó khi update. Cách này khớp tuyệt đối và dùng **mật độ thật**.
- Jacobian tính bằng công thức ổn định số học: $\log(1-\tanh^2u) = 2(\log 2 - u - \text{softplus}(-2u))$.
- Đường `action=` (u = atanh(clamp(action))) chỉ giữ lại cho evaluator và demo.
- Lưới an toàn:
  - kẹp `log_ratio` trong ±20;
  - bỏ qua minibatch actor nếu loss không hữu hạn, đồng thời ghi lại số minibatch bị bỏ (`skipped_actor_updates`) và in cảnh báo `[NaN GUARD]`.

### Bài học cho báo cáo
- Một bản vá "làm hai đường khớp nhau" vẫn có thể sai nếu chọn sai điểm để khớp. Test chỉ kiểm tra sự khớp nhau (TEST 1B) không phát hiện được lỗi này. Cần test đo trực tiếp cơ chế (TEST 4D: dịch mean và so sánh ratio).
- **Resume**: tiếp tục từ `best_agent.pt` (step 1.37M, lưu trước khi nổ). Metrics có mốc `resume_marks` ghi kèm commit mới. Đoạn 0 → 1.37M vẫn là bằng chứng hợp lệ cho F1.

---

## F14. σ ga/phanh kẹt trần dù đã có projection (entropy trên Gaussian trước tanh)

### Hiện tượng [Đã kiểm chứng: `logs/p1_main`, checkpoint]
- **Lần 1**: σ_throttle chạm 1.0 ở khoảng 370k step và nằm ở đó tới 1.37M; σ_brake ở 1.0 trong đoạn 330k–700k.
- **Lần 2**: sau khi resume F13 (1.39M), σ_brake tăng đều từ 0.22 lên 1.0 ở khoảng 1.8M và kẹt ở đó. Checkpoint lúc ấy có logstd = [−1.57, **0, 0**].
- Tỉ lệ eval sạch (crash 0%, completion ≥ 98%) giảm dần:

| Đoạn step | Tỉ lệ eval sạch |
| :--- | :--- |
| 1.40–1.69M | 48% |
| 1.69–2.00M | 30% |
| 2.00–2.20M | **0%** |

- Apex accuracy giảm từ khoảng 75% xuống khoảng 45%.

### Cơ chế [Suy luận]
- Entropy của Gaussian trước tanh là $\sum \log\sigma + c$. Bonus `ent_coef` đẩy σ đi lên với một lực **không đổi**, ở mọi state.
- **Ở vùng bão hòa**: đạp hết ga trên đoạn thẳng có $\mu_u \gg 0$, không phanh có $\mu_u \ll 0$. Lúc đó nhiễu trong $u$ hầu như không làm đổi hành động (tanh phẳng), nên policy gradient **không phạt** σ. Entropy thắng, σ lên trần.
- **Ở vùng phanh** (mean gần ngưỡng deadband): σ = 1 khiến phanh trong rollout gần như ngẫu nhiên. Agent không học được điểm phanh, nên crash ngày càng nhiều.
- Khác với F1: lần này gradient vẫn còn sống (có projection), chỉ là entropy thắng. Lỗi gần giống F1 nhưng nhẹ hơn.

### Can thiệp và kết quả [Đã kiểm chứng]
- **Resume lần 2** tại 2.31M từ `best_agent` (khoảng 1.86M): `--ent-coef 0.001`, đồng thời reset logstd ga/phanh về −1.0 (σ 0.37), giữ nguyên σ lái.
- **σ ga/phanh**: giảm dần 0.36 → 0.27, tức policy gradient điều khiển được σ trở lại.
- **σ lái**: **sập về sàn 0.10 trong khoảng 150k step**. Đây là tác dụng phụ của ent 0.001.
- **Best lap sạch**: 130.5s → **125.6s** (ở khoảng 2.45M). Tỉ lệ eval sạch phục hồi lên 27% trong đoạn 2.40–2.51M.

### Bài học
- `ent_coef` = 0.01 cố định làm σ ga/phanh kẹt trần.
- `ent_coef` = 0.001 cố định làm σ lái sập về sàn.
- Hai chiều hành động cần **chế độ khám phá khác nhau**, nên Phase 2 dùng `LOGSTD_MAX` theo từng chiều, cộng với entropy decay (D19).

---

## F15. Thói quen lái: chậm vì thả trôi, không phải vì lạng lách

**[Đã kiểm chứng: 1 vòng deterministic, xuất phát đứng, map nhỏ]** Script tham chiếu: `scratch/drive_trace_reference.py`.

| Checkpoint | Lap | TB / max km/h | Full ga | Thả trôi | Có phanh | Đảo lái /100 bước |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| P1 khoảng 1.55M | 137.2s | 73.8 / 94.6 | 12% | 59% | 17% | 30.8 |
| P1 best khoảng 1.86M | 133.9s | 75.9 / 102.1 | 19% | 50% | 15% | 34.1 |
| P1 best khoảng 2.45M | 130.1s | 78.1 / 102.4 | 22% | 48% | 16% | 36.1 |
| P1 2500k | crash ở 21.7s | 60.3 / 100.1 | 56% | 24% | 10% | 35.5 |
| **111s best** | **115.1s** | **88.4 / 121.1** | **73%** | **1%** | **32%** | **38.9** |

- **Lạng lách không phải nguyên nhân**: agent 111s còn đảo lái nhiều hơn. *Lưu ý (I10)*: chỉ số đảo lái thấp hơn của P1 một phần là do P1 chạy chậm và không tấn công cua, nên không được diễn giải thành "P1 lái mượt hơn".
- Agent P1 **giảm tốc bằng nhả ga thay vì phanh**, và chưa bao giờ lên được trần 120 km/h. Theo thời gian nó đạp ga nhiều dần lên (12% → 22% → 56%), nhưng phanh vẫn ít (10–17%, so với 32% ở 111s). Vì vậy khi bạo ga hơn thì nó crash ở cua đầu, đúng kiểu crash "khoảng 5% vòng" thấy trong eval.
- **Giả thuyết (chưa chứng minh)**:
  1. 1.37M step đầu học với estimator bị lỗi F13 cho đúng hành động đạp hết ga.
  2. F14 khiến phanh trong rollout là nhiễu.
  3. Bias −1.6 làm agent ít khám phá phanh.

---

## F16. Ma sát cầu trước chưa bao giờ được áp; xe chổng đuôi khi phanh mạnh

**[Đã kiểm chứng 30/10]** Cầu trước luôn có μ = 0.9 do `attach_agent` ghi ma sát qua list bánh lỗi thời (cả bản 111s và Phase 1); phanh > ~1.7 G làm xe chổng đuôi. Toàn bộ phân tích, số đo và cách xử lý: `knowledge/vehicle_and_engine_physics.md` §2.1, §4, §5.

---

## Tổng hợp insight Phase 1 → quyết định Phase 2

| # | Insight | Áp dụng cho Phase 2 |
| :-- | :--- | :--- |
| I1 | F1: projection có hiệu quả. Lần đầu có vòng sạch ở khoảng 1.35M, trong khi run cũ chạy 3.38M không có vòng sạch nào | Giữ |
| I2 | F13: log_prob tính tại $u$ đã sample | Giữ; smoke test bắt buộc có TEST 4D |
| I3 | F14: σ ga/phanh kẹt trần khi ent = 0.01 | D19: `LOGSTD_MAX` theo từng chiều |
| I4 | ent = 0.001 làm σ lái sập về sàn trong khoảng 150k | D19: entropy decay, không dùng 0.001 ngay từ đầu |
| I5 | Lap time bị giới hạn bởi cách dùng ga/phanh | `scripts/drive_trace.py` là công cụ theo dõi chuẩn |
| I6 | Phanh ít được khám phá | Theo dõi % phanh của P2; nếu thấp kéo dài thì xem lại bias |
| I7 | Eval 3 episode gần như giống hệt nhau, kết quả nhảy giữa 0% và 100% | P2 dùng `--eval-episodes 5` |
| I8 | P1 cần khoảng 2.5M step để xuống khoảng 126s; map 1:1 tốn khoảng ×1.44 số bước mỗi vòng | 6M cho P2-single là vừa đủ, không dư |
| I9 | Sau resume lần 2 (ent 0.001), σ lái về sàn 0.10 và **eval crash ở đúng một chỗ (46%, khoảng mét 1306, khúc cua rộng) suốt 8 lần eval liên tiếp**; train không còn về đích. Value loss rất thấp (0.05–0.2) vì critic đã dự đoán đúng rằng sẽ crash ở đó, nên không còn tín hiệu khám phá. Ở một nhánh cũ, dùng decay lr/ent thì giai đoạn sau cũng gần như không cải thiện | Không decay; ent 0.01 cố định cộng với trần σ theo từng chiều; giảm entropy bằng fine-tune theo đoạn (D22) |
| I10 | Quan sát demo: P1 không vượt quá khoảng 100 km/h, vào cua nhẹ và phanh nhẹ hơn bản 111s, có 1–2 chỗ giật lái gây drift. **Độ mượt tay lái thấp hơn của P1 là ảo giác thống kê**, do agent không tấn công cua chứ không phải lái khéo hơn | Khi báo cáo, so độ mượt tay lái phải chuẩn hóa theo tốc độ hoặc theo cua |
| I11 | Phạt lắc tay lái `−0.1·Δ²` lớn nhất chỉ khoảng −0.003/bước (do giới hạn tốc độ xoay 0.18), so với tiến độ khoảng 0.15/bước, nên giật lái gần như miễn phí (cả ở bản 111s). Reward tiến độ tính theo hình chiếu lên tim đường thưởng nhẹ cho việc ở phía trong cua ($v\cdot R/(R-lat)$) | Ghi là giới hạn; chưa sửa vì chưa có bằng chứng giật lái gây crash |
| I12 | **Bản 111s nhanh hơn ghost** (eval 111.4s so với T* 112.6s), và vẫn chạy hết vòng trong env hiện tại với luật 4 bánh (115.1s, xuất phát đứng) | Trong báo cáo gọi ghost là **"tham chiếu giải tích"**, không gọi là cận trên. Lý do bị vượt: các bậc $a_{lat}$ khớp tay thận trọng và giới hạn lệch ngang ±4.05m. Cần đo thêm vòng flying của bản 111s trong env hiện tại. Phân tích sai số mô hình đầy đủ: `knowledge/vehicle_and_engine_physics.md` §7 |

---

## Ma trận ablation đề xuất

Cùng seed, mỗi run 500k step. Nếu đủ thời gian, chạy 2–3 seed. Metric theo dõi:
- σ từng chiều theo thời gian
- completion, crash_rate
- best lap

| ID | Cấu hình | Câu hỏi trả lời |
| :-- | :--- | :--- |
| A0 | Code trước vá (log sẵn có: 42D, hiện tại) | Baseline lỗi. Lưu ý obs khác nhau nên không so sánh sạch |
| A1 | A0 + projection logstd (F1) | F1 có đủ để chống kẹt σ không? |
| A2 | A1 + ACTION_BOUND (F2) | Tác động của việc log_prob khớp nhau |
| A3 | A2 + bias phanh −1.6 (F3) | Tốc độ học giai đoạn đầu |
| A4 | A3 + time penalty 0.01 (F7) | Hiệu ứng lên lap time |
| A5 | A4 với thưởng về đích hàm mũ thay cho η³ (F6) | Độ dốc tín hiệu lap time |
| A6 | A5 + sector bonus K=10, thay thế bonus ở đích (F6, F11) | Credit assignment cục bộ |

**Lịch thực tế (plan v7.1)**:
- P1-main = A5, chạy 4M step.
- A0 = P1-main với `--no-logstd-projection`, chạy 1M step.
- A1 lấy từ 1M step đầu của P1-main, nên không cần chạy riêng.
- P1-sector = A6.
- Các run A2–A4 bỏ qua vì không đủ thời gian.

Ưu tiên nếu thiếu thời gian:
- A0 so với A4: lỗi so với bản vá đầy đủ.
- A1 so với A0: bằng chứng cô lập cho F1, đây là phát hiện mạnh nhất.
