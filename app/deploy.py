"""Run the private API and public Streamlit UI in one deployment service."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Sequence
from urllib.error import URLError
from urllib.request import urlopen


def _start(command: Sequence[str], *, env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(command, env=env)


def _wait_for_api(process: subprocess.Popen[bytes], url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"API exited during startup with code {process.returncode}")
        try:
            with urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.25)
    raise TimeoutError(f"API did not become ready at {url} within {timeout:g} seconds")


def _stop(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    """Supervise both processes and stop the service if either one exits."""

    api_port = os.environ.get("API_PORT", "8000")
    public_port = os.environ.get("PORT", "10000")
    api_url = f"http://127.0.0.1:{api_port}"
    environment = os.environ.copy()
    environment["HR_API_URL"] = api_url

    processes: list[subprocess.Popen[bytes]] = []
    try:
        api = _start(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                api_port,
            ],
            env=environment,
        )
        processes.append(api)
        _wait_for_api(api, f"{api_url}/health")

        ui = _start(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app/streamlit_app.py",
                "--server.address",
                "0.0.0.0",
                "--server.port",
                public_port,
                "--browser.gatherUsageStats",
                "false",
                "--server.headless",
                "true",
            ],
            env=environment,
        )
        processes.append(ui)

        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        return next(
            (process.returncode or 0 for process in processes if process.poll() is not None),
            1,
        )
    except KeyboardInterrupt:
        return 130
    finally:
        _stop(processes)


if __name__ == "__main__":
    raise SystemExit(main())
