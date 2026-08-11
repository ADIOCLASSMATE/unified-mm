#!/usr/bin/env python3
"""Run Accelerate with a numeric loopback port check.

The Ascend Job image does not resolve the hostname ``localhost``. Accelerate
1.14 checks the rendezvous port through that hostname before it starts
torchrun, so multi-node launch otherwise fails before any worker is created.
Keep Accelerate's check intact while using the equivalent numeric loopback
address, without mutating the container's /etc/hosts or vendored packages.
"""

from __future__ import annotations

import socket
from typing import Optional


def numeric_loopback_port_in_use(port: Optional[int] = None) -> bool:
    if port is None:
        port = 29500
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def main() -> None:
    import accelerate.utils.launch as launch_utils
    from accelerate.commands.accelerate_cli import main as accelerate_main

    launch_utils.is_port_in_use = numeric_loopback_port_in_use
    accelerate_main()


if __name__ == "__main__":
    main()
