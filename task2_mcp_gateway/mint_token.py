"""Mint a demo JWT. Usage: python task2_mcp_gateway/mint_token.py admin|viewer [--ttl 3600]"""

from __future__ import annotations

import argparse
import os
import time

import jwt


def mint(role: str, secret: str, ttl: int = 3600, subject: str = "demo-user") -> str:
    now = int(time.time())
    claims = {"sub": subject, "role": role, "aud": "mcp-gateway", "iat": now, "exp": now + ttl}
    return jwt.encode(claims, secret, algorithm="HS256")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["admin", "viewer"])
    parser.add_argument("--ttl", type=int, default=3600)
    args = parser.parse_args()
    secret = os.environ.get("MCP_GATEWAY_JWT_SECRET")
    if not secret:
        raise SystemExit("Set MCP_GATEWAY_JWT_SECRET first")
    print(mint(args.role, secret, args.ttl))
