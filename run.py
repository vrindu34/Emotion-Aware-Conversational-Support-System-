#!/usr/bin/env python3
"""
run.py
──────
Convenience launcher — starts FastAPI backend and Streamlit frontend
in separate processes.

Usage:
    python run.py                  # start both
    python run.py --backend-only   # API only
    python run.py --frontend-only  # UI only (assumes API is running)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent


def start_backend():
    print("▶ Starting FastAPI backend on http://localhost:8000")
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
        ],
        cwd=ROOT,
    )


def start_frontend():
    print("▶ Starting Streamlit frontend on http://localhost:8501")
    return subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            str(ROOT / "frontend" / "app.py"),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
        ],
        cwd=ROOT,
    )


def main():
    parser = argparse.ArgumentParser(description="Launch the Emotion Support System")
    parser.add_argument("--backend-only", action="store_true")
    parser.add_argument("--frontend-only", action="store_true")
    args = parser.parse_args()

    processes = []
    try:
        if not args.frontend_only:
            processes.append(start_backend())
        if not args.backend_only:
            import time; time.sleep(2)   # let backend boot
            processes.append(start_frontend())

        print("\n✅ System running:")
        if not args.frontend_only:
            print("   API docs  → http://localhost:8000/docs")
            print("   API base  → http://localhost:8000/api")
        if not args.backend_only:
            print("   Chat UI   → http://localhost:8501")
        print("\nPress Ctrl+C to stop.\n")

        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        print("\n⏹ Shutting down...")
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
