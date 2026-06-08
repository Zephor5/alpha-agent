"""Local URL safety checks for read-only web tools.

This is a local preflight gate for caller-supplied URLs. When another service
performs the final fetch, its DNS resolution and redirect handling can still
differ from this process, so this module reduces risk but does not prove the
remote fetch is completely SSRF-safe.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterable
from urllib.parse import urlsplit

Resolver = Callable[[str], Iterable[str]]

_ALLOWED_SCHEMES = {"http", "https"}
_CLOUD_METADATA_HOSTS = {
    "metadata.google.internal",
    "metadata.goog",
}
_CLOUD_METADATA_IPS = frozenset(
    ipaddress.ip_address(address)
    for address in (
        "169.254.169.254",
        "169.254.170.2",
        "169.254.169.253",
        "fd00:ec2::254",
        "100.100.100.200",
    )
)
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


def validate_public_http_url(url: str, *, resolver: Resolver | None = None) -> None:
    """Fail closed unless *url* is an http(s) URL resolving only to public IPs."""

    parsed = urlsplit(str(url or "").strip())
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError("URL scheme must be http/https")
    if "@" in parsed.netloc:
        raise ValueError("URL userinfo is not allowed")
    host = parsed.hostname
    if not host:
        raise ValueError("URL host is required")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL port is invalid") from exc
    del port

    normalized_host = _normalize_host(host)
    if normalized_host in _CLOUD_METADATA_HOSTS:
        raise ValueError("URL host targets a cloud metadata service")

    literal_ip = _parse_ip_address(normalized_host)
    if literal_ip is not None:
        _ensure_allowed_ip(literal_ip)
        return

    addresses = _resolve_addresses(normalized_host, resolver=resolver)
    if not addresses:
        raise ValueError("DNS resolution failed for URL host")
    for address in addresses:
        _ensure_allowed_ip(address)


def _normalize_host(host: str) -> str:
    return host.strip().rstrip(".").lower()


def _parse_ip_address(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def _resolve_addresses(
    host: str,
    *,
    resolver: Resolver | None,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        raw_addresses = tuple(resolver(host) if resolver is not None else _socket_resolver(host))
    except socket.gaierror as exc:
        raise ValueError("DNS resolution failed for URL host") from exc
    except OSError as exc:
        raise ValueError("DNS resolution failed for URL host") from exc

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for raw_address in raw_addresses:
        address = _parse_ip_address(str(raw_address))
        if address is None:
            raise ValueError("DNS resolution failed for URL host")
        addresses.append(address)
    return tuple(dict.fromkeys(addresses))


def _socket_resolver(host: str) -> tuple[str, ...]:
    infos = socket.getaddrinfo(host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
    addresses: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr:
            addresses.append(str(sockaddr[0]))
    return tuple(dict.fromkeys(addresses))


def _ensure_allowed_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    comparable: ipaddress.IPv4Address | ipaddress.IPv6Address = address
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        comparable = address.ipv4_mapped
    if comparable in _CLOUD_METADATA_IPS or address in _CLOUD_METADATA_IPS:
        raise ValueError("URL address targets a cloud metadata service and is not allowed")
    if isinstance(comparable, ipaddress.IPv4Address) and comparable in _CGNAT_NETWORK:
        raise ValueError("URL address is in CGNAT space and is not allowed")
    if (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        raise ValueError("URL address is not allowed")
    if comparable is not address and (
        comparable.is_loopback
        or comparable.is_private
        or comparable.is_link_local
        or comparable.is_reserved
        or comparable.is_multicast
        or comparable.is_unspecified
    ):
        raise ValueError("URL address is not allowed")
