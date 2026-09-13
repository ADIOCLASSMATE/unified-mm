"""Read-only guard for the explicitly direct image/Qwen network paths."""

from pathlib import Path
import ssl


def direct_ssl_context():
    """Verified TLS with a compact key share for this environment's direct egress.

    The local OpenSSL 3.5 default handshake times out on the SII path, while
    P-256 succeeds. Keep certificate/hostname verification and TLS 1.3 support;
    do not change routes, proxy services, or the agent's connection settings.
    """
    context = ssl.create_default_context()
    context.set_ecdh_curve("prime256v1")
    return context


def check_direct_routes():
    route = Path("/proc/net/route")
    if not route.exists():
        raise RuntimeError("cannot inspect Linux routes for direct network access")
    devices = {line.split()[0] for line in route.read_text().splitlines()[1:] if line.split()}
    ipv6 = Path("/proc/net/ipv6_route")
    if ipv6.exists():
        devices.update(line.split()[-1] for line in ipv6.read_text().splitlines() if line.split())
    tunnels = sorted(d for d in devices if d.startswith(("tun", "tap", "wg", "tailscale", "ppp"))
                     or (Path("/sys/class/net") / d / "tun_flags").exists())
    if tunnels:
        raise RuntimeError(f"direct network access requires routes without tunnel interfaces: {tunnels}")
    return sorted(devices)
