#!/usr/bin/env python3
"""
setup_ollama.py
───────────────
One-shot bootstrap that:
  1. Detects or installs Ollama on the host.
  2. Ensures the Ollama daemon is running.
  3. Pulls qwen2.5:1.5b and qwen2.5:3b if not already present.

Run once before launching the agent:
    python setup_ollama.py
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request
import urllib.error

# ── SSL fix ───────────────────────────────────────────────────────────────────
# macOS Python installers do NOT link to the system CA bundle.
# We bypass cert verification for the Ollama installer fetch only.
_NO_VERIFY_CTX = ssl.create_default_context()
_NO_VERIFY_CTX.check_hostname = False
_NO_VERIFY_CTX.verify_mode    = ssl.CERT_NONE


def _urlopen(url: str, timeout: int = 60):
    return urllib.request.urlopen(url, timeout=timeout, context=_NO_VERIFY_CTX)


# ─── config ───────────────────────────────────────────────────────────────────

MODEL_NAMES = [
    os.environ.get("STELLAV_REASON_1_5B", "qwen2.5:1.5b"),
    os.environ.get("STELLAV_REASON_3B", "qwen2.5:3b"),
]
MODEL_NAME = MODEL_NAMES[0]
OLLAMA_BASE  = "http://localhost:11434"
MAX_WAIT_SEC = 120

# macOS puts the CLI here after the .app installer runs
MACOS_FALLBACK_PATHS = [
    "/usr/local/bin/ollama",
    "/opt/homebrew/bin/ollama",
    os.path.expanduser("~/.ollama/ollama"),
]


def _find_ollama_bin() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    for p in MACOS_FALLBACK_PATHS:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ollama_alive() -> bool:
    try:
        _urlopen(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _wait_for_ollama() -> bool:
    print(f"[setup] Waiting for Ollama daemon (up to {MAX_WAIT_SEC}s)…", flush=True)
    for _ in range(MAX_WAIT_SEC):
        if _ollama_alive():
            print("[setup] Ollama is up.", flush=True)
            return True
        time.sleep(1)
    return False


# ─── install ──────────────────────────────────────────────────────────────────

def install_ollama() -> str:
    """
    Install Ollama and return the path to the CLI binary.

    On macOS the official install.sh downloads the .app bundle and drops the
    CLI at /usr/local/bin/ollama.  The script ends with `open -a Ollama` which
    fails when there is no GUI session — that's what produced the
    "Unable to find application named 'Ollama'" error.  We intentionally
    ignore a non-zero exit from the installer and just verify the binary
    landed on disk instead.
    """
    system = platform.system().lower()

    if system == "windows":
        print(
            "[setup] Windows detected.\n"
            "        Please install Ollama manually: https://ollama.com/download\n"
            "        then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[setup] Ollama not found — downloading installer…", flush=True)
    try:
        with _urlopen("https://ollama.com/install.sh", timeout=60) as resp:
            script = resp.read().decode()
    except Exception as exc:
        print(f"[setup] Could not fetch installer: {exc}", file=sys.stderr)
        sys.exit(1)

    print("[setup] Running installer (this may take a minute)…", flush=True)
    proc = subprocess.run(["bash", "-s"], input=script, text=True, check=False)

    # The installer exits non-zero on macOS when `open -a Ollama` fails
    # (no GUI / Launch Services not yet updated).  That's fine — the CLI
    # binary is still installed.  We check for the binary explicitly.
    bin_path = _find_ollama_bin()
    if bin_path:
        print(f"[setup] Ollama CLI found at {bin_path}", flush=True)
        return bin_path

    # If the binary is genuinely missing, bail out with a helpful message.
    print(
        "[setup] Installer finished but the 'ollama' binary could not be found.\n"
        "        Tried: " + ", ".join(MACOS_FALLBACK_PATHS) + "\n"
        "        Please install manually: https://ollama.com/download",
        file=sys.stderr,
    )
    sys.exit(1)


# ─── daemon ───────────────────────────────────────────────────────────────────

def ensure_daemon(ollama_bin: str) -> None:
    if _ollama_alive():
        print("[setup] Ollama daemon already running.", flush=True)
        return

    print("[setup] Starting Ollama daemon in the background…", flush=True)
    subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if not _wait_for_ollama():
        print(
            "[setup] ERROR: Ollama daemon did not start.\n"
            "        Try running manually in another terminal:  ollama serve",
            file=sys.stderr,
        )
        sys.exit(1)


# ─── model pull ───────────────────────────────────────────────────────────────

def _listed_models() -> list[str]:
    try:
        with _urlopen(f"{OLLAMA_BASE}/api/tags", timeout=10) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _model_present(name: str, listed: list[str] | None = None) -> bool:
    names = listed if listed is not None else _listed_models()
    return any(name == n or n.startswith(name + ":") or name in n for n in names)


def ensure_model(ollama_bin: str) -> None:
    listed = _listed_models()
    for name in MODEL_NAMES:
        if _model_present(name, listed):
            print(f"[setup] Model '{name}' already present.", flush=True)
            continue
        print(f"[setup] Pulling '{name}' — this may take several minutes…", flush=True)
        subprocess.run([ollama_bin, "pull", name], check=True)
        print(f"[setup] '{name}' ready.", flush=True)


# ─── entry ────────────────────────────────────────────────────────────────────

def main() -> None:
    bin_path = _find_ollama_bin()

    if not bin_path:
        bin_path = install_ollama()

    # Make sure the binary directory is on PATH for the rest of this session
    bin_dir = os.path.dirname(bin_path)
    if bin_dir and bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    ensure_daemon(bin_path)
    ensure_model(bin_path)

    print("\n[setup] Ready. Selectable reasoners: 1.5b -> qwen2.5:1.5b, 3b -> qwen2.5:3b.", flush=True)
    print("[setup] Run:  python main.py ask \"what is a randomized trial?\" --reason 1.5b", flush=True)


if __name__ == "__main__":
    main()
