"""
NAVAL-SEM Desktop Launcher
Starts the FastAPI backend on a free port, then opens the UI in a browser window.
Packaged by PyInstaller into a single .exe or .msi.
"""

import os
import sys
import threading
import time
import webbrowser
import socket
import signal
import logging

# ── Silence noisy loggers in packaged mode ────────────────────────────────────
logging.basicConfig(level=logging.WARNING)

# ── Resolve base path (works both in dev and in PyInstaller bundle) ────────────
def resource_path(relative: str) -> str:
    """Return absolute path to a resource, works for dev and PyInstaller."""
    if getattr(sys, "_MEIPASS", None):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), relative)


# ── Find a free TCP port ───────────────────────────────────────────────────────
def find_free_port(start: int = 8765) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found in range 8765–8865")


# ── Start the FastAPI/Uvicorn server in a daemon thread ───────────────────────
def start_server(port: int):
    import uvicorn
    # Tell the app module where static files live
    os.environ["NAVAL_SEM_STATIC"] = resource_path("static")
    os.environ["NAVAL_SEM_PORT"] = str(port)
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )


# ── Wait until server is accepting connections ────────────────────────────────
def wait_for_server(port: int, timeout: float = 15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


# ── Try to use pywebview for a native window, fall back to browser ─────────────
def open_ui(url: str):
    try:
        import webview  # pip install pywebview
        webview.create_window(
            "NAVAL-SEM",
            url,
            width=1200,
            height=820,
            resizable=True,
            min_size=(900, 600),
        )
        webview.start()
    except ImportError:
        # pywebview not available → use system browser
        webbrowser.open(url)
        # Keep the process alive so the server keeps running
        try:
            signal.pause()          # Unix
        except AttributeError:
            input("NAVAL-SEM is running. Press Enter to quit...\n")
    except Exception as e:
        print(f"[NAVAL-SEM] Could not open webview: {e}. Falling back to browser.")
        webbrowser.open(url)
        try:
            signal.pause()
        except AttributeError:
            input("NAVAL-SEM is running. Press Enter to quit...\n")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    port = find_free_port()
    print(f"[NAVAL-SEM] Starting server on port {port}...")

    server_thread = threading.Thread(
        target=start_server, args=(port,), daemon=True
    )
    server_thread.start()

    if not wait_for_server(port):
        print("[NAVAL-SEM] ERROR: Server did not start within 15 seconds.")
        sys.exit(1)

    url = f"http://127.0.0.1:{port}"
    print(f"[NAVAL-SEM] Ready → {url}")
    open_ui(url)


if __name__ == "__main__":
    main()
