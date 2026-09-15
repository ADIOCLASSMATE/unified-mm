#!/usr/bin/env python3
"""Refresh the existing Inspire account with CAS-only proxy routing.

Run through script/inspire_wjx_login.sh to use the installed Inspire runtime.
Only this process changes Requests routing; global proxy settings stay intact.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from urllib.parse import urlsplit

import requests


@contextmanager
def cas_proxy_route(proxy):
    original = requests.Session.send

    def send(session, request, **kwargs):
        host = urlsplit(request.url).hostname
        if host == "cas.sii.edu.cn":
            kwargs["proxies"] = {"http": proxy, "https": proxy}
        elif host in {"qz.sii.edu.cn", "keycloak-inspire-prod.sii.edu.cn"}:
            # Requests can carry the CAS proxy into the redirect despite NO_PROXY.
            kwargs["proxies"] = {}
        return original(session, request, **kwargs)

    requests.Session.send = send
    try:
        yield
    finally:
        requests.Session.send = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", default="wjx-ascend")
    args = parser.parse_args()
    from inspire.platform.web.session.auth import _login_with_cas_requests, get_credentials, _load_runtime_config
    from inspire.platform.web.session.proxy import resolve_requests_proxy_config

    proxies, _ = resolve_requests_proxy_config(args.account)
    proxy = proxies.get("https") or proxies.get("http") or proxies.get("all")
    if not proxy:
        parser.error("CAS routing requires the existing shell/account HTTPS proxy; run without unsetting proxy variables")
    username, password = get_credentials(args.account)
    try:
        with cas_proxy_route(proxy):
            session = _login_with_cas_requests(username, password,
                base_url=_load_runtime_config(args.account).base_url, account=args.account)
    except Exception as exc:
        # Raw auth exceptions may include session-bearing redirect URLs.
        print(json.dumps({"login": "failed", "account": args.account, "error_type": type(exc).__name__}))
        raise SystemExit(1) from None
    print(json.dumps({"login": "success", "account": args.account,
                      "workspaces_discovered": len(session.all_workspace_ids or []),
                      "user_detail_available": bool(session.user_detail)}))


if __name__ == "__main__":
    main()
