from __future__ import annotations

import socket
from collections.abc import Iterable

import pytest

from alpha_agent.tools.url_safety import validate_public_http_url


def _resolver(*addresses: str):
    def resolve(_host: str) -> Iterable[str]:
        return addresses

    return resolve


def test_validate_public_http_url_allows_public_http_hosts() -> None:
    validate_public_http_url("https://example.com/article", resolver=_resolver("1.1.1.1"))
    validate_public_http_url(
        "http://example.com:8080/article",
        resolver=_resolver("2606:4700:4700::1111"),
    )


def test_validate_public_http_url_allows_public_literal_ip_without_dns() -> None:
    def resolver(_host: str) -> Iterable[str]:
        raise AssertionError("literal IPs should not use DNS")

    validate_public_http_url("https://1.1.1.1/article", resolver=resolver)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "//example.com/path",
    ],
)
def test_validate_public_http_url_rejects_non_http_schemes(url: str) -> None:
    with pytest.raises(ValueError, match="http/https"):
        validate_public_http_url(url, resolver=_resolver("1.1.1.1"))


@pytest.mark.parametrize(
    "url",
    [
        "https://user@example.com/article",
        "https://user:password@example.com/article",
    ],
)
def test_validate_public_http_url_rejects_userinfo(url: str) -> None:
    with pytest.raises(ValueError, match="userinfo"):
        validate_public_http_url(url, resolver=_resolver("1.1.1.1"))


@pytest.mark.parametrize("url", ["https:///article", "https://"])
def test_validate_public_http_url_rejects_empty_host(url: str) -> None:
    with pytest.raises(ValueError, match="host is required"):
        validate_public_http_url(url, resolver=_resolver("1.1.1.1"))


@pytest.mark.parametrize("url", ["https://example.com:abc", "https://example.com:99999"])
def test_validate_public_http_url_rejects_invalid_port_before_dns(url: str) -> None:
    def resolver(_host: str) -> Iterable[str]:
        raise AssertionError("invalid ports should be rejected before DNS")

    with pytest.raises(ValueError, match="port is invalid"):
        validate_public_http_url(url, resolver=resolver)


def test_validate_public_http_url_fails_closed_on_dns_failure() -> None:
    def resolver(_host: str) -> Iterable[str]:
        raise socket.gaierror("no address")

    with pytest.raises(ValueError, match="DNS resolution failed"):
        validate_public_http_url("https://example.invalid/article", resolver=resolver)


def test_validate_public_http_url_fails_closed_on_empty_dns_result() -> None:
    with pytest.raises(ValueError, match="DNS resolution failed"):
        validate_public_http_url("https://example.invalid/article", resolver=_resolver())


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.0.1/",
        "http://169.254.1.1/",
        "http://169.254.169.254/",
        "http://169.254.170.2/",
        "http://169.254.169.253/",
        "http://100.100.100.200/",
        "http://100.64.0.1/",
        "http://0.0.0.0/",
        "http://224.0.0.1/",
        "http://240.0.0.1/",
        "http://[::1]/",
        "http://[fc00::1]/",
        "http://[fe80::1]/",
        "http://[ff02::1]/",
        "http://[fd00:ec2::254]/",
    ],
)
def test_validate_public_http_url_rejects_unsafe_literal_ips(url: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        validate_public_http_url(url, resolver=_resolver("1.1.1.1"))


@pytest.mark.parametrize(
    "url",
    [
        "https://metadata.google.internal/computeMetadata/v1",
        "https://METADATA.GOOG./computeMetadata/v1",
    ],
)
def test_validate_public_http_url_always_rejects_cloud_metadata_hosts(url: str) -> None:
    def resolver(_host: str) -> Iterable[str]:
        raise AssertionError("metadata hosts should be rejected before DNS")

    with pytest.raises(ValueError, match="cloud metadata"):
        validate_public_http_url(url, resolver=resolver)


@pytest.mark.parametrize("address", ["10.0.0.1", "100.64.0.1", "169.254.169.254"])
def test_validate_public_http_url_rejects_unsafe_resolved_addresses(address: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        validate_public_http_url(
            "https://public-name.example/article",
            resolver=_resolver("1.1.1.1", address),
        )
