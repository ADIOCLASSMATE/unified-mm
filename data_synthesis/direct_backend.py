"""Per-connection TCP MSS set BEFORE SYN; never alters host routes or proxies."""
import asyncio
import socket

import anyio
import httpcore
from httpcore._backends.anyio import AnyIOBackend, AnyIOStream


class DirectMSSBackend(AnyIOBackend):
    def __init__(self, mss):
        if not 256 <= mss <= 1400:
            raise ValueError("direct TCP MSS must be between 256 and 1400")
        self.mss = mss

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        loop = asyncio.get_running_loop()
        try:
            async with asyncio.timeout(timeout):
                addresses = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
                last_error = None
                for family, kind, proto, _, address in addresses:
                    sock = socket.socket(family, kind, proto)
                    try:
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, self.mss)
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        for option in socket_options or ():
                            sock.setsockopt(*option)
                        sock.setblocking(False)
                        if local_address:
                            sock.bind((local_address, 0))
                        await loop.sock_connect(sock, address)
                        return AnyIOStream(await anyio.abc.SocketStream.from_socket(sock))
                    except OSError as exc:
                        last_error = exc
                        sock.close()
                    except BaseException:
                        sock.close()
                        raise
                raise last_error or OSError("no direct address resolved")
        except TimeoutError as exc:
            raise httpcore.ConnectTimeout(str(exc)) from exc
        except OSError as exc:
            raise httpcore.ConnectError(str(exc)) from exc
