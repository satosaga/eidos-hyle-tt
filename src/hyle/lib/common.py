#!/usr/bin/env python3
"""
HYLE - common.py

Small shared helpers for the HYLE utility series. Not a standalone tool;
imported by hyle_*.py entry points that need a consistent "drop a file on
this window" launch experience (prompt_via_drag_drop) or a consistent local
HTTP server launch/shutdown experience (QuietHTTPHandler, run_local_server).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_TITLE = "HYLE^TT"


def window_title(app_name: str) -> str:
    """Return the standard '{PROJECT_TITLE} {app_name}' window/tab title.

    Single source of truth for the "HYLE^TT " prefix, so it can't drift out
    of sync (missing the "^TT", wrong case, snake_case module name leaking
    through, ...) between the tkinter drag-drop window and the browser-tab
    <title> each hyle.apps.* tool otherwise sets independently.
    """
    return f"{PROJECT_TITLE} {app_name}"


def prompt_via_drag_drop(
    message: str,
    win_title: str = PROJECT_TITLE,
) -> Path | None:
    """Open a small always-on-top window that accepts a dragged-and-dropped
    file, and return its path.

    Used instead of a directory-based file picker because HYLE input files
    (strategy JSONs, FIT files, ...) end up scattered across ``resources/strategies/*/``,
    ``resources/exports/*/*/``, and ad-hoc copies -- there's no single directory a picker
    could sensibly default to. Drag-and-drop from Finder (search results,
    any nested folder, wherever) sidesteps that entirely.

    Returns None if the window was closed without dropping anything.

    Requires tkinterdnd2 (pip install tkinterdnd2); exits with a clear
    install instruction if it's missing rather than silently falling back
    to a lesser experience.
    """
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
    except ImportError:
        sys.exit(
            "error: tkinterdnd2 is required for the drag-and-drop picker:\n\n"
            "    pip install tkinterdnd2\n"
        )

    import tkinter as tk

    dropped: dict[str, Path] = {}

    root = TkinterDnD.Tk()
    root.title(win_title)
    root.geometry("380x180")
    root.attributes("-topmost", True)  # otherwise it's easy to lose behind Finder

    label = tk.Label(
        root,
        text=message,
        font=("Helvetica", 14),
        relief="groove",
        borderwidth=2,
    )
    label.pack(expand=True, fill="both", padx=16, pady=16)

    def on_drop(event):
        # event.data can be brace-quoted (paths with spaces) and/or contain
        # multiple files; tk's own splitlist handles both correctly.
        paths = root.tk.splitlist(event.data)
        if paths:
            dropped["path"] = Path(paths[0])
            root.destroy()

    # tkinterdnd2 monkey-patches these methods onto every widget at runtime
    # (via TkinterDnD.Tk()'s root), so they don't exist in tkinter's own
    # stubs -- not a real type error.
    label.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
    label.dnd_bind("<<Drop>>", on_drop)  # type: ignore[attr-defined]

    root.mainloop()
    return dropped.get("path")


class QuietHTTPHandler(BaseHTTPRequestHandler):
    """BaseHTTPRequestHandler with request logging silenced and a small
    `_send` response helper. Shared by the hyle.apps.* tools that run a
    local "serve one page, talk to it over HTTP" server
    (hyle.apps.fit2gpx_converter, hyle.apps.cpmodel_estimator).

    Subclass and implement `do_GET` / `do_POST` as usual. If the tool needs
    the standard HYLE "browser tab closed -> stop the server" convention
    (POST /shutdown, triggered by navigator.sendBeacon() on the page's
    pagehide event), call `self._handle_shutdown()` from do_POST when
    `path == "/shutdown"` instead of reimplementing it.
    """

    def log_message(self, fmt, *args):  # noqa: A003 -- quiet by default
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_shutdown(self) -> None:
        """Standard /shutdown handling: acknowledge immediately, then stop
        the server from a background thread so this response isn't blocked
        by serve_forever()'s own loop exiting (ThreadingHTTPServer already
        runs this handler off the main thread, so there's no deadlock risk
        either way)."""
        self._send(200, b"ok", "text/plain")
        threading.Thread(target=self.server.shutdown, daemon=True).start()


class _FailFastHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer, except an application bug raised while handling
    a request kills the whole process instead of socketserver's own
    default handle_error() (print a traceback, keep serving other
    requests) -- matches this project's fail-fast default (see
    core/__init__.py's np.seterr/sys.excepthook; neither of those apply
    here, since a request is handled in its own thread, inside
    socketserver's own try/except, well before either hook would ever
    see the exception).

    Client-side connection drops (BrokenPipeError/ConnectionResetError/
    ConnectionAbortedError) are excluded on purpose: a browser tab
    closing or reloading mid-request triggers these routinely and
    they're not application bugs -- those still just get logged the
    default way, exactly as before this class existed.
    """

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            super().handle_error(request, client_address)
            return
        traceback.print_exc()
        # os._exit() skips normal interpreter shutdown (buffered
        # stdout/stderr flushing included) -- without this, output
        # written just before the crash can be silently lost. See
        # core/__init__.py's _fail_fast_excepthook for the same fix.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def run_local_server(
    handler_class: type[BaseHTTPRequestHandler],
    serving_message: str,
    extra_messages: list[str] | None = None,
) -> None:
    """Start a _FailFastHTTPServer on 127.0.0.1 (random free port), open the
    browser to it, and block until either Ctrl+C or the page's own
    /shutdown route (see QuietHTTPHandler._handle_shutdown) stops it.

    `serving_message` is a format string that receives `url` (e.g.
    "serving {url}"); `extra_messages` are printed as-is right after it,
    before the standard "closing the browser tab stops this..." line.
    """
    server = _FailFastHTTPServer(("127.0.0.1", 0), handler_class)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"

    logger.info(serving_message.format(url=url))
    for msg in extra_messages or []:
        logger.info(msg)
    logger.info("closing the browser tab stops this automatically (or press Ctrl+C)")

    webbrowser.open(url, new=1)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopped")
    else:
        # serve_forever() returned on its own -- the /shutdown handler
        # called server.shutdown(), triggered by the browser tab closing.
        logger.info("stopped (browser tab closed)")
