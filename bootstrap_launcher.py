from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


def app_root() -> Path:
    # The launcher is copied beside the source/scripts folder for the test kit.
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


class BootstrapWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Voice Identity Studio - Setup")
        self.root.geometry("700x440")
        self.root.resizable(False, False)
        self.log = tk.Text(root, height=18, width=84, state="disabled")
        self.log.pack(padx=14, pady=(14, 8))
        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(fill="x", padx=14, pady=4)
        buttons = tk.Frame(root)
        buttons.pack(pady=10)
        self.cpu_button = tk.Button(buttons, text="Cài CPU", width=18, command=lambda: self.start(False))
        self.cpu_button.pack(side="left", padx=5)
        self.gpu_button = tk.Button(buttons, text="Cài GPU NVIDIA", width=18, command=lambda: self.start(True))
        self.gpu_button.pack(side="left", padx=5)
        tk.Button(buttons, text="Mở app", width=18, command=self.launch, state="disabled").pack(side="left", padx=5)
        self.launch_button = buttons.winfo_children()[-1]
        if self.ready():
            self.write("Runtime và model đã sẵn sàng. Đang mở app...")
            self.root.after(400, self.launch)
        else:
            self.write("Chọn CPU hoặc GPU để cài runtime và tải model lần đầu.")

    def ready(self) -> bool:
        root = app_root()
        runtime = root / ".runtime" / "Scripts" / "python.exe"
        models = root / "models"
        return runtime.exists() and any(models.rglob("campplus_cn_en_common.pt")) and any(models.rglob("encoder-*.onnx"))

    def write(self, value: str) -> None:
        self.root.after(0, self._write, value)

    def _write(self, value: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", value + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self, gpu: bool) -> None:
        self.cpu_button.configure(state="disabled")
        self.gpu_button.configure(state="disabled")
        self.launch_button.configure(state="disabled")
        self.progress.start(12)
        threading.Thread(target=self._bootstrap, args=(gpu,), daemon=True).start()

    def _bootstrap(self, gpu: bool) -> None:
        script = app_root() / "scripts" / "bootstrap_test.ps1"
        if not script.exists():
            self.write(f"Không tìm thấy: {script}")
            self.root.after(0, lambda: self.progress.stop())
            return
        command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
        if gpu:
            command.append("-Gpu")
        self.write("Đang cài runtime và tải model, vui lòng chờ...")
        process = subprocess.Popen(command, cwd=app_root(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            self.write(line.rstrip())
        code = process.wait()
        self.root.after(0, self.progress.stop)
        if code == 0:
            self.write("Hoàn tất. Bấm 'Mở app'.")
            self.root.after(0, lambda: self.launch_button.configure(state="normal"))
        else:
            self.write(f"Cài đặt thất bại, mã lỗi {code}.")
            self.cpu_button.configure(state="normal")
            self.gpu_button.configure(state="normal")

    def launch(self) -> None:
        script = app_root() / "scripts" / "run_test.ps1"
        if not script.exists():
            messagebox.showerror("Thiếu file", f"Không tìm thấy {script}")
            return
        subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)], cwd=app_root())


if __name__ == "__main__":
    window = tk.Tk()
    BootstrapWindow(window)
    window.mainloop()
