"""
NAVAL-SEM Desktop Launcher
Starts the FastAPI backend on a free port, then opens the UI in a native
pywebview window (or falls back to the system browser if unavailable).
Packaged by PyInstaller into a single .exe.
"""

import os
import sys
import threading
import time
import webbrowser
import socket
import logging

logging.basicConfig(level=logging.WARNING)


# ── Resolve base path ────────────────────────────────────────────────────────
def resource_path(relative: str) -> str:
    if getattr(sys, "_MEIPASS", None):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), relative)


# ── Find a free TCP port ─────────────────────────────────────────────────────
def find_free_port(start: int = 8765) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found in range 8765-8865")


# ── Start FastAPI/Uvicorn in a daemon thread ──────────────────────────────────
def start_server(port: int):
    import uvicorn
    os.environ["NAVAL_SEM_STATIC"] = resource_path("static")
    os.environ["NAVAL_SEM_PORT"] = str(port)
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )


# ── Wait until server accepts connections ─────────────────────────────────────
def wait_for_server(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.15)
    return False


# ── Open UI ───────────────────────────────────────────────────────────────────
def open_ui(url: str):
    # Try pywebview first (native desktop window)
    try:
        import webview
        webview.create_window(
            "NAVAL-SEM",
            url,
            width=1200,
            height=820,
            resizable=True,
            min_size=(900, 600),
        )
        webview.start()
        return  # webview.start() blocks until window is closed — clean exit
    except ImportError:
        pass  # pywebview not installed — fall through to browser
    except Exception:
        pass  # pywebview failed (e.g. WebView2 missing) — fall through

    # Fallback: open in system browser and keep server alive via Event
    webbrowser.open(url)
    _keep_alive()


def _keep_alive():
    """Keep the server process alive when running in browser-fallback mode.
    Uses threading.Event instead of input() so it works with console=False."""
    stop = threading.Event()

    def _watch():
        # Exit when the server thread dies (shouldn't happen, but safety net)
        while threading.active_count() > 1:
            time.sleep(1)
        stop.set()

    threading.Thread(target=_watch, daemon=True).start()
    stop.wait()  # blocks forever until the server dies or process is killed


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    port = find_free_port()

    server_thread = threading.Thread(
        target=start_server, args=(port,), daemon=True
    )
    server_thread.start()

    if not wait_for_server(port, timeout=20):
        # Can't show console message with console=False — try browser anyway
        webbrowser.open(f"http://127.0.0.1:{port}")
        _keep_alive()
        return

    url = f"http://127.0.0.1:{port}"
    open_ui(url)


if __name__ == "__main__":
    main()
