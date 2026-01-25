!git clone  https://github.com/JAKEee3e/Manga-Colorizer.git
!pip install Flask-Cors einops
!pip install Flask-Cors
!pip install einops
!pip install --upgrade pip
!pip install opencv-python-headless
import os
import subprocess
import threading
import sys
import warnings

# Suppress PyTorch SourceChangeWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*SourceChangeWarning.*")

# ------------------------
# Paths
# ------------------------
BACKEND_DIR = "/teamspace/studios/this_studio/Manga-Colorizer/Backend"  # adjust if needed
os.chdir(BACKEND_DIR)

# ------------------------
# Run backend
# ------------------------
from collections import deque
import socket

python_process = None
backend_log_tail = deque(maxlen=120)


def stop_backend_if_running():
    global python_process
    if python_process is not None and python_process.poll() is None:
        print("[*] Stopping previous backend process...")
        try:
            python_process.terminate()
            python_process.wait(timeout=10)
        except Exception:
            try:
                python_process.kill()
            except Exception:
                pass


def pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


stop_backend_if_running()

PORT = pick_free_port()
print(f"[*] Using backend port: {PORT}")


def run_backend():
    global python_process, backend_log_tail

    # T4 GPU Optimizations (16GB VRAM) - MAXIMUM SPEED MODE:
    # --fp16: CRITICAL for T4 - halves memory usage, enables 1.5-2x speedup
    # --compile: OPTIONAL - can be unstable with dynamic shapes / concurrent requests
    # --max-image-size 1280: Balanced speed/quality (faster than 1536px, better than 1024px)
    command = [sys.executable, "app-stream.py", "--no-ssl", "--fp16", "--max-image-size", "1280", "--port", str(PORT)]
    # If you want to try compile anyway:
    # command = [sys.executable, "app-stream.py", "--no-ssl", "--fp16", "--compile", "--max-image-size", "1280", "--port", str(PORT)]

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # Suppress PyTorch SourceChangeWarning
    env["PYTHONWARNINGS"] = "ignore::UserWarning:torch.serialization"

    python_process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=BACKEND_DIR,
        env=env
    )

    for line in python_process.stdout:
        line = line.rstrip("\n")
        # Skip PyTorch SourceChangeWarning messages
        if "SourceChangeWarning" in line:
            continue
        backend_log_tail.append(line)
        print(line)

    code = python_process.poll()
    print(f"[!] Backend process exited with code {code}. Last logs:\n" + "\n".join(list(backend_log_tail)[-60:]))


backend_thread = threading.Thread(target=run_backend, daemon=True)
backend_thread.start()

# Wait a moment for backend to start
import time
import socket
time.sleep(2)  # Give backend thread a moment to start

# Wait for backend server to be ready
print("[*] Waiting for backend server to start (this may take 30-60 seconds while models load)...")
max_wait = 120
waited = 0
backend_ready = False

while waited < max_wait:
    # If the backend process crashed/exited, fail fast and show captured logs
    if python_process is not None:
        code = python_process.poll()
        if code is not None:
            tail = "\n".join(list(backend_log_tail)[-60:])
            raise RuntimeError(f"Backend process exited early with code {code}. Last logs:\n{tail}")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', PORT))
        sock.close()
        if result == 0:
            # Double-check: try to actually connect and get a response
            try:
                import urllib.request
                response = urllib.request.urlopen(f'http://127.0.0.1:{PORT}/', timeout=2)
                if response.getcode() == 200:
                    backend_ready = True
                    print("[+] Backend server is ready!")
                    break
            except:
                pass  # Not quite ready yet
    except Exception:
        pass

    time.sleep(3)  # Check every 3 seconds
    waited += 3
    if waited % 15 == 0:
        print(f"[*] Still waiting for backend... ({waited}s/{max_wait}s) - Models may still be loading")

if not backend_ready:
    print("[-] Backend server didn't start in time. Check backend logs above for errors.")
    print("[*] Common issues:")
    print("    - Model files missing or corrupted")
    print("    - CUDA/GPU initialization errors")
    print("    - Port 5000 already in use")
    raise TimeoutError("Backend server failed to start within 120 seconds")

# Small delay to ensure Flask is fully ready
time.sleep(2)

# Use Cloudflare Tunnel (cloudflared) - free, no auth required
print("[*] Creating public URL with Cloudflare Tunnel (cloudflared)...")
import re

def ensure_cloudflared():
    try:
        subprocess.run(["cloudflared", "--version"], capture_output=True, timeout=5, check=True)
        return "cloudflared"
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        # Check if it exists in user's local bin
        local_bin = os.path.expanduser("~/.local/bin/cloudflared")
        if os.path.exists(local_bin) and os.access(local_bin, os.X_OK):
            return local_bin
        # Check if it exists in /tmp
        tmp_path = "/tmp/cloudflared"
        if os.path.exists(tmp_path) and os.access(tmp_path, os.X_OK):
            return tmp_path
        return None

cloudflared_path = ensure_cloudflared()

if not cloudflared_path:
    print("[*] Installing cloudflared...")
    local_bin_dir = os.path.expanduser("~/.local/bin")
    os.makedirs(local_bin_dir, exist_ok=True)
    cloudflared_path = os.path.join(local_bin_dir, "cloudflared")
    
    subprocess.run([
        "wget", "-q",
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        "-O", cloudflared_path
    ], check=True)
    subprocess.run(["chmod", "+x", cloudflared_path], check=True)
    print(f"[*] Installed cloudflared to {cloudflared_path}")

tunnel_proc = subprocess.Popen(
    [cloudflared_path, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

public_url = None
for _ in range(60):
    if tunnel_proc.poll() is not None:
        raise RuntimeError("cloudflared tunnel exited early")
    line = tunnel_proc.stdout.readline()
    if line:
        print(f"[*] {line.rstrip()}")
        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if m:
            public_url = m.group(0)
            break
    time.sleep(0.5)

if public_url:
    print(f"\n[+] Manga-Colorizer is live at: {public_url}")
    print(f"[*] Tunnel PID: {tunnel_proc.pid}")
else:
    print("[!] Could not detect tunnel URL. Check output above for trycloudflare.com URL.")
    print(f"[*] Or run manually: {cloudflared_path} tunnel --url http://127.0.0.1:{PORT}")
