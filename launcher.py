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
from pathlib import Path


# ── Log file setup ────────────────────────────────────────────────────────────
def setup_logging():
    """Write logs to platform-appropriate app data directory."""
    try:
        if sys.platform == "win32":
            base = Path(os.environ.get("APPDATA", Path.home()))
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
        log_dir = base / "NAVAL-SEM"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "naval_sem.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
            ]
        )
        logging.info("NAVAL-SEM starting up")
        logging.info(f"Log file: {log_file}")
        logging.info(f"Python: {sys.version}")
        logging.info(f"Frozen: {getattr(sys, 'frozen', False)}")
        return log_file
    except Exception:
        # If logging setup fails, fall back to silent mode
        logging.basicConfig(level=logging.WARNING)
        return None


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
    logging.info(f"Starting uvicorn on port {port}")
    try:
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            log_config=None,  # disable uvicorn log config — avoids isatty() crash with console=False
        )
    except Exception as e:
        logging.error(f"Server crashed: {e}", exc_info=True)


# ── Wait until server accepts connections ─────────────────────────────────────
def wait_for_server(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                logging.info(f"Server ready on port {port}")
                return True
        except OSError:
            time.sleep(0.15)
    logging.error(f"Server did not start within {timeout}s")
    return False


# ── Open UI ───────────────────────────────────────────────────────────────────
def open_ui(url: str):
    # Try pywebview first (native desktop window)
    try:
        import webview
        logging.info("Opening pywebview window")
        webview.create_window(
            "NAVAL-SEM",
            url,
            width=1200,
            height=820,
            resizable=True,
            min_size=(900, 600),
        )
        webview.start()
        logging.info("pywebview window closed — exiting")
        return
    except ImportError:
        logging.warning("pywebview not available — falling back to browser")
    except Exception as e:
        logging.warning(f"pywebview failed ({e}) — falling back to browser")

    # Fallback: open in system browser
    logging.info(f"Opening browser at {url}")
    webbrowser.open(url)
    _keep_alive()


def _keep_alive():
    """Keep process alive in browser-fallback mode without console input."""
    stop = threading.Event()

    def _watch():
        while threading.active_count() > 1:
            time.sleep(1)
        stop.set()

    threading.Thread(target=_watch, daemon=True).start()
    stop.wait()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    log_file = setup_logging()

    try:
        port = find_free_port()
        logging.info(f"Using port {port}")

        server_thread = threading.Thread(
            target=start_server, args=(port,), daemon=True
        )
        server_thread.start()

        if not wait_for_server(port, timeout=20):
            logging.error("Server failed to start — opening browser anyway")
            webbrowser.open(f"http://127.0.0.1:{port}")
            _keep_alive()
            return

        url = f"http://127.0.0.1:{port}"
        open_ui(url)

    except Exception as e:
        logging.critical(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
