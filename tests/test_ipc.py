"""Single-instance handshake over QLocalServer/QLocalSocket."""

from __future__ import annotations

import time
import uuid

from PySide6.QtCore import QCoreApplication

from shotfinder.ipc import SingleInstance, default_server_name


def _application() -> QCoreApplication:
    return QCoreApplication.instance() or QCoreApplication([])


def _unique_name() -> str:
    return f"shotfinder-test-{uuid.uuid4().hex}"


def _wait_for(app: QCoreApplication, predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def test_second_launch_notifies_the_running_instance() -> None:
    app = _application()
    name = _unique_name()
    running = SingleInstance(name)
    received: list[str] = []
    running.messageReceived.connect(received.append)
    try:
        assert running.listen()
        second = SingleInstance(name)

        assert second.notify_running(2000) is True
        assert _wait_for(app, lambda: bool(received))
        assert received == ["show"]
    finally:
        running.close()


def test_no_listener_means_this_is_the_first_instance() -> None:
    instance = SingleInstance(_unique_name())

    assert instance.notify_running(300) is False


def test_listener_close_releases_the_name() -> None:
    name = _unique_name()
    first = SingleInstance(name)
    assert first.listen()
    first.close()

    second = SingleInstance(name)
    try:
        assert second.listen()
    finally:
        second.close()


def test_default_name_is_stable_and_user_scoped() -> None:
    assert default_server_name() == default_server_name()
    assert default_server_name().startswith("ShotFinder-")
    assert default_server_name("Other") != default_server_name()
