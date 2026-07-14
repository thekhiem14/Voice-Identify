# Voice Identity Studio

Ứng dụng desktop chạy local để phân đoạn, định danh người nói và nhận dạng tiếng
Việt từ một file hội thoại.

```text
audio + voice samples
  -> mono 16 kHz
  -> DiariZen diarization
  -> remap cluster + merge gap < 2 s
  -> clean 10 s cluster enrollment
  -> ERes2Net cosine verification (threshold 0.40)
  -> majority vote + hard override
  -> Gipformer Vietnamese ASR
  -> result.json
```

## Cài đặt

Yêu cầu: Windows, Python 3.11 được khuyến nghị, Git, FFmpeg và khoảng trống đủ
cho checkpoint. CUDA được tự động sử dụng nếu bản PyTorch đang cài hỗ trợ GPU.

```powershell
cd meeting-insight-app
Copy-Item .env.example .env
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

### NVIDIA RTX GPU trên Windows

Nếu môi trường cũ đã cài PyTorch CPU, đóng mọi cửa sổ đang chạy `python app.py`, rồi chạy:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_gpu.ps1
```

Script cài `torch`/`torchaudio` CUDA 12.8, dependency khử nhiễu và kiểm tra
`torch.cuda.is_available()` trước khi báo thành công. Cấu hình mặc định dùng `cuda:0`,
DiariZen batch size 2 cho GPU 6 GB và Gipformer yêu cầu provider `cuda`.

Console in tiến độ DiariZen, ERes2Net và Gipformer mỗi 100 chunk/segment. Có thể
đổi tần suất bằng `PROGRESS_LOG_EVERY` trong `.env`.

Setup tải/cài mã nguồn chính thức của
[DiariZen](https://github.com/BUTSpeechFIT/DiariZen),
[3D-Speaker/ERes2Net](https://github.com/modelscope/3D-Speaker) và checkpoint
[Gipformer](https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt).

Nếu đã có model và chỉ muốn cài dependency:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -SkipModels
python scripts/download_models.py --only eres2net
```

## Chạy giao diện

```powershell
python app.py
```

Luồng sử dụng nhanh:

1. Chọn audio cuộc họp.
2. Nhập tên, chọn một hoặc nhiều sample rồi bấm **Thêm người**. Có thể bỏ qua
   nếu đã lưu các người nói trong tab **Kho mẫu giọng**.
3. Tùy chọn nhập `1=0` để gộp cluster hoặc `2=Hưng` để ép tên cluster.
4. Bấm **Chạy pipeline**. Mỗi lần chạy tạo một job riêng trong `outputs/`.

Sample không bắt buộc. Khi không có profile dùng được, app vẫn chạy DiariZen và
Gipformer, sau đó đặt tên tạm `Speaker 1`, `Speaker 2`... theo cluster.

Sample chỉ dùng cho lần hiện tại không ghi vào database. Tab **Kho mẫu giọng**
cho phép lưu embedding để dùng lại ở những lần sau.

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
  --threshold 0.40
```

Lưu mẫu vào database:

```powershell
python app.py enroll `
  --speaker-id hung `
  --display-name "Hưng" `
  --samples .\samples\hung_1.wav .\samples\hung_2.wav

python app.py profiles
```

Smoke test chỉ diarization, không tải ERes2Net/Gipformer:

```powershell
python app.py run --audio .\meeting.wav --skip-identify --skip-asr
```

## Output

Mỗi job thành công chỉ gồm 3 file:

- `transcript.json`: output chính, tối giản `segments[{speaker,start,end,text}]`.
- `transcript.txt`: biên bản dễ mở bằng Notepad.
- `result.json`: cảnh báo, cosine, cluster và thời gian chạy dành cho kiểm tra kỹ thuật.

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
  Các đoạn diarization dài cũng được chia thành cửa sổ identity/ASR tối đa 8 giây
  để giảm nguy cơ embedding trung bình che mất một lần đổi người nói.
- Không có sample: kết quả mang trạng thái `diarization_only` và tên tạm
  `Speaker 1...k`. Hard override như `2=Hưng` vẫn hoạt động.
- Có sample nhưng cosine dưới `0.40`: nhãn cuối là `Unknown`, kèm
  `provisional_speaker` và bảng `scores` để kiểm tra.
- Sample thiếu, dưới 1 giây, im lặng, embedding hỏng hoặc sai dimension: sample
  đó bị bỏ qua; profile khác vẫn chạy và warning được ghi vào JSON.
- Audio im lặng hoặc dưới 0,25 giây: không gọi DiariZen, trả timeline rỗng cùng
  warning thay vì hallucinate cluster.
- ASR lỗi ở một segment: segment đó có `asr_status=error`; các segment khác vẫn
  được xuất. Nếu Gipformer hoặc ERes2Net không khởi tạo được, mặc định app xuất
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
- `core/voice_id.py`: ERes2Net, enrollment, cosine verification.
- `core/asr_engine.py`: Gipformer qua sherpa-onnx.
- `ui/main_layout.py`: giao diện Flet.
- `config/settings.py`: model ID, threshold và đường dẫn.

## Kiểm thử

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q -p no:cacheprovider tests
```

Ngưỡng `0.40` là giá trị mặc định hiện tại, không phải ngưỡng tối
ưu cho mọi micro/phòng họp. Nên hiệu chỉnh trên tập validation thực tế trước khi
dùng cho quyết định nhạy cảm.
