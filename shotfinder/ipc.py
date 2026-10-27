"""Single-instance handshake over a local socket.

The first launch listens on a per-user named pipe; a second launch connects to
it, asks the running app to show its search window and exits. Qt's
``QLocalServer``/``QLocalSocket`` wrap the platform primitives (named pipes on
Windows), so no extra dependency or port is involved.
"""

from __future__ import annotations

import getpass
import hashlib
import logging
import os
import time
from typing import Callable

from PySide6.QtCore import QCoreApplication, QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

log = logging.getLogger(__name__)
SHOW_MESSAGE = "show"
_CONNECT_TIMEOUT_MS = 1500


def default_server_name(app_name: str = "ShotFinder") -> str:
    """A pipe name that is stable per user and does not collide between users."""
    user = os.environ.get("USERNAME") or getpass.getuser()
    digest = hashlib.sha256(f"{app_name}:{user}".encode("utf-8")).hexdigest()[:16]
    return f"{app_name}-{digest}"


class SingleInstance(QObject):
    """Listen for a second launch, or notify the instance that already runs."""

    messageReceived = Signal(str)

    def __init__(self, name: str | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._name = name or default_server_name()
        self._server: QLocalServer | None = None
        self._handler: Callable[[str], None] | None = None

    @property
    def name(self) -> str:
        return self._name

    def notify_running(self, timeout_ms: int = _CONNECT_TIMEOUT_MS) -> bool:
        """Ask the running instance to show its window.

        Returns ``True`` when a server accepted the message, so the caller
        should exit; ``False`` means this process is the first instance.
        """
        socket = QLocalSocket()
        socket.connectToServer(self._name)
        if not socket.waitForConnected(timeout_ms):
            socket.abort()
            return False
        socket.write(f"{SHOW_MESSAGE}\n".encode("utf-8"))
        socket.flush()
        # QLocalSocket's blocking writer does not flush this small payload
        # reliably before the process exits, so pump the event loop until the
        # bytes have left the Qt buffer and are in the pipe.
        application = QCoreApplication.instance()
        deadline = time.monotonic() + timeout_ms / 1000.0
        while socket.bytesToWrite() > 0 and time.monotonic() < deadline:
            if application is not None:
                application.processEvents()
            time.sleep(0.005)
        socket.waitForBytesWritten(50)
        socket.disconnectFromServer()
        if socket.state() != QLocalSocket.LocalSocketState.UnconnectedState:
            socket.waitForDisconnected(timeout_ms)
        return True

    def listen(self, on_message: Callable[[str], None] | None = None) -> bool:
        """Start accepting handshakes; returns ``False`` if the pipe is taken."""
        if self._server is not None:
            return True
        self._handler = on_message
        # A crashed process can leave the pipe behind; removing it is safe
        # because notify_running() has already ruled out a live server.
        QLocalServer.removeServer(self._name)
        server = QLocalServer(self)
        server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        server.newConnection.connect(self._accept_connections)
        if not server.listen(self._name):
            log.warning(
                "Could not listen on %s: %s", self._name, server.errorString()
            )
            server.deleteLater()
            return False
        self._server = server
        return True

    def close(self) -> None:
        server = self._server
        self._server = None
        self._handler = None
        if server is None:
            return
        server.close()
        QLocalServer.removeServer(self._name)
        server.deleteLater()

    def _accept_connections(self) -> None:
        server = self._server
        if server is None:
            return
        while server.hasPendingConnections():
            socket = server.nextPendingConnection()
            if socket is None:
                break
            socket.readyRead.connect(lambda sock=socket: self._read(sock))
            socket.disconnected.connect(socket.deleteLater)
            if socket.bytesAvailable():
                self._read(socket)

    def _read(self, socket: QLocalSocket) -> None:
        payload = bytes(socket.readAll()).decode("utf-8", errors="replace").strip()
        if not payload:
            return
        self.messageReceived.emit(payload)
        if self._handler is not None:
            self._handler(payload)
