# Voice Identity Studio

Ứng dụng desktop chạy local để phân đoạn, định danh người nói và nhận dạng tiếng
Việt từ một file hội thoại.

```text
audio + voice samples
  -> mono 16 kHz
  -> DiariZen diarization
  -> remap cluster + merge gap <= 2 s (tối đa 20 s)
  -> clean 10 s cluster enrollment
  -> CAM++ cosine verification trên cửa sổ <= 8 s (threshold 0.33)
  -> majority vote + hard override
  -> Gipformer Vietnamese ASR trên segment merge, bỏ segment < 1.5 s
  -> result.json
```

Trong `result.json`, `segments` là timeline đã merge dùng cho ASR; `identity_windows`
giữ bằng chứng CAM++ chi tiết theo cửa sổ để không làm mất thông tin voice identify.
Gipformer luôn đọc bản mono 16 kHz chưa khử nhiễu, kể cả khi bật enhancement, vì
denoise có thể làm méo thanh điệu tiếng Việt; phần preprocess/enhancement vẫn được
giữ cho các tầng diarization và voice identify.

## Cài đặt

Yêu cầu: Windows, Python 3.11 được khuyến nghị, Git, FFmpeg và khoảng trống đủ
cho checkpoint. Bộ cài mặc định dùng PyTorch CPU để chạy được cả trên máy không
có NVIDIA.

```powershell
cd meeting-insight-app
Copy-Item .env.example .env
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

### Bản test tải runtime sau khi cài

Có thể gửi source app (không kèm `models/` và `.runtime/`) cho người test. Máy
test chạy lệnh dưới đây để tạo Python runtime riêng, cài dependency và tải
checkpoint vào máy lần đầu:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_test.ps1
```

Máy có NVIDIA dùng thêm `-Gpu`. Sau khi bootstrap xong, chạy app bằng:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_test.ps1
```

Runtime được lưu trong `.runtime/`, còn model trong `models/`; các lần sau
không phải cài/tải lại. Cách này cần máy test có Python và Git, đồng thời có
Internet ở lần cài đầu tiên.

### NVIDIA RTX GPU trên Windows

Nếu môi trường cũ đã cài PyTorch CPU, đóng mọi cửa sổ đang chạy `python app.py`, rồi chạy:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_gpu.ps1
```

Script GPU nâng `torch`/`torchaudio` lên CUDA 12.8, cài DLL sherpa-onnx CUDA và
kiểm tra `torch.cuda.is_available()`. Trong UI, công tắc **Dùng GPU cho toàn
pipeline** chuyển đồng bộ DiariZen, enhancement, CAM++ và Gipformer giữa hai mode:

- Tắt: toàn bộ inference chạy CPU; DiariZen dùng batch 1 để giới hạn RAM.
- Bật: toàn bộ inference chạy CUDA; Gipformer vẫn decode batch 1 để tránh lỗi
  attention encoder cấp phát hơn 1 GB VRAM.

CAM++ ghép tối đa 16 cửa sổ có cùng chính xác độ dài feature trong một batch
(`VOICE_ID_BATCH_SIZE=16`), nên không cần padding và không làm lệch embedding.

Console in tiến độ DiariZen, CAM++ và Gipformer mỗi 100 chunk/segment. Có thể
đổi tần suất bằng `PROGRESS_LOG_EVERY` trong `.env`.

Setup tải/cài mã nguồn chính thức của
[DiariZen](https://github.com/BUTSpeechFIT/DiariZen),
[3D-Speaker/CAM++](https://github.com/modelscope/3D-Speaker), checkpoint
[CAM++ Chinese-English Advanced](https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced) và
[Gipformer](https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt).

Nếu đã có model và chỉ muốn cài dependency:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -SkipModels
python scripts/download_models.py --only campplus
```

## Chạy giao diện

```powershell
python app.py
```

Luồng sử dụng nhanh:

1. Chọn audio cuộc họp.
2. Nhập tên, chọn một hoặc nhiều sample rồi bấm **Thêm người**. Có thể bỏ qua
   nếu đã lưu các người nói trong tab **Kho mẫu giọng**.
3. Chọn CPU hoặc bật **Dùng GPU cho toàn pipeline**. Máy chưa chạy
   `scripts/install_gpu.ps1` phải để CPU.
4. Tùy chọn nhập `1=0` để gộp cluster hoặc `2=Hưng` để ép tên cluster.
5. Bấm **Chạy pipeline**. Mỗi lần chạy tạo một job riêng trong `outputs/`.

Console ghi từng stage theo dạng `[TIMING][CPU] diarization=...` hoặc
`[TIMING][GPU] asr=...`, rồi ghi tổng thời gian và RTF khi hoàn tất. UI hiển thị
cùng số liệu; nếu chạy cùng một file lần lượt bằng CPU và GPU trong cùng phiên,
app tự hiện tỷ lệ nhanh/chậm tổng và cho từng stage. RTF càng thấp càng nhanh.
Timing bao gồm thời gian tải model; nên chạy mỗi mode hai lần trên cùng file và
so các lượt sau để giảm sai lệch cold-start/cache.

Sample không bắt buộc. Khi không có profile dùng được, app vẫn chạy DiariZen và
Gipformer, sau đó đặt tên tạm `Speaker 1`, `Speaker 2`... theo cluster.

Sample chỉ dùng cho lần hiện tại không ghi vào database. Tab **Kho mẫu giọng**
cho phép lưu embedding để dùng lại ở những lần sau.
Branch CAM++ dùng riêng `data/database/voice_db_campplus.json`. Embedding ERes2Net
cũ không tương thích, vì vậy cần enroll lại từ các file sample giọng gốc; database
ERes2Net cũ vẫn được giữ nguyên.

## CLI

Chạy một lần với sample truyền trực tiếp:

```powershell
python app.py run `
  --audio .\meeting.wav `
  --sample "Hưng=.\samples\hung_1.wav" `
  --sample "Hưng=.\samples\hung_2.wav" `
  --sample "B=.\samples\b.wav" `
  --cluster-map "1=0" `
  --force-speaker "2=Hưng" `
  --threshold 0.33
```

Lưu mẫu vào database:

```powershell
python app.py enroll `
  --speaker-id hung `
  --display-name "Hưng" `
  --samples .\samples\hung_1.wav .\samples\hung_2.wav

python app.py profiles
```

Smoke test chỉ diarization, không tải CAM++/Gipformer:

```powershell
python app.py run --audio .\meeting.wav --skip-identify --skip-asr
```

## Output

Mỗi job thành công chỉ gồm 3 file:

- `transcript.json`: output chính, tối giản `segments[{speaker,start,end,text}]`.
- `transcript.txt`: biên bản dễ mở bằng Notepad.
- `result.json`: cảnh báo, cosine, cluster và thời gian chạy dành cho kiểm tra kỹ thuật.

`result.json.metrics` gồm `runtime_mode`, `runtime_seconds`,
`audio_duration_seconds`, `realtime_factor`, `audio_seconds_per_runtime_second`
và `step_runtime_seconds`, đủ để so sánh lại ngoài UI.

`failure.json` chỉ xuất hiện nếu pipeline thất bại. Audio chuẩn hóa, audio khử nhiễu,
mẫu cluster và các file trong `work/` đều là tạm thời và được tự động xóa sau khi chạy.

Mỗi segment trong `result.json` giữ cả nhãn cuối và bằng chứng trước smoothing:

```json
{
  "segment_id": "seg_0003",
  "cluster": "SPEAKER_02",
  "start": 12.44,
  "end": 16.71,
  "speaker": "Hưng",
  "score": 0.7214,
  "source": "majority_vote",
  "raw_speaker": "Hưng",
  "raw_score": 0.7042,
  "text": "nội dung nhận dạng tiếng Việt"
}
```

## Các trường hợp ngoại lệ

`n` là số profile dùng được, `k` là số cluster do DiariZen tạo. Hai giá trị này
không bắt buộc bằng nhau vì profile có thể là danh sách người *có khả năng* xuất
hiện, không phải danh sách người chắc chắn có mặt trong audio.

- `k > n`: app cho phép nhiều cluster cùng mang một identity. Trường
  `split_identity_candidates` và warning `possible_over_clustering` cho biết
  DiariZen có thể đã tách một người thành nhiều cluster.
- `k < n`: profile không xuất hiện được liệt kê ở `profiles_not_observed`. Nếu
  cùng một cluster có đủ bằng chứng của nhiều người, cluster được đánh dấu
  `mixed_cluster`; app giữ identity từng segment thay vì ép cả cluster theo vote.
  Các đoạn diarization dài cũng được chia thành cửa sổ identity tối đa 8 giây
  để giảm nguy cơ embedding trung bình che mất một lần đổi người nói.
- Không có sample: kết quả mang trạng thái `diarization_only` và tên tạm
  `Speaker 1...k`. Hard override như `2=Hưng` vẫn hoạt động.
- Có sample nhưng cosine dưới `0.33`: nhãn cuối là `Unknown`, kèm
  `provisional_speaker` và bảng `scores` để kiểm tra.
- Sample thiếu, dưới 1 giây, im lặng, embedding hỏng hoặc sai dimension: sample
  đó bị bỏ qua; profile khác vẫn chạy và warning được ghi vào JSON.
- Audio im lặng hoặc dưới 0,25 giây: không gọi DiariZen, trả timeline rỗng cùng
  warning thay vì hallucinate cluster.
- ASR lỗi ở một segment: segment đó có `asr_status=error`; các segment khác vẫn
  được xuất. Nếu Gipformer hoặc CAM++ không khởi tạo được, mặc định app xuất
  partial result. Dùng `--strict` nếu muốn dừng toàn bộ job.
- Mapping/override trỏ tới cluster không tồn tại hoặc mapping hai chiều: không
  làm job crash; output có warning cấu hình.

Các warning có cấu trúc `{code, severity, message, ...context}`. Trường
`reconciliation` trong `result.json` luôn ghi số profile, số cluster, cluster
trộn, profile chưa quan sát và identity bị tách qua nhiều cluster.

## Cấu trúc chính

- `core/pipeline.py`: điều phối đầy đủ 8 bước và ghi output.
- `core/diarization.py`: adapter DiariZen.
- `core/segment_processing.py`: remap, merge, loại overlap, vote và override.
- `core/voice_id.py`: CAM++, enrollment, cosine verification.
- `core/asr_engine.py`: Gipformer qua sherpa-onnx.
- `ui/main_layout.py`: giao diện Flet.
- `config/settings.py`: model ID, threshold và đường dẫn.

## Kiểm thử

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q -p no:cacheprovider tests
```

Ngưỡng `0.33` là giá trị mặc định do checkpoint CAM++ Chinese-English công bố,
không phải ngưỡng tối
ưu cho mọi micro/phòng họp. Nên hiệu chỉnh trên tập validation thực tế trước khi
dùng cho quyết định nhạy cảm.
