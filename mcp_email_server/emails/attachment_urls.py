"""Resolution of URL-supplied attachments for compose workflows.

``send_email``, ``save_to_mailbox``, and ``save_draft`` accept attachment
references. Local filesystem paths remain the primary form; this module adds
URL references — ``http://``/``https://`` for remote files and ``data:`` for
inline base64 payloads — by materializing them into a caller-managed
temporary directory. Existing path-based validation, preflight, and MIME
composition then operate on the materialized files unchanged.

Security model, mirroring ``_preflight_attachment_sizes`` in
``application/mutations.py``: URL resolution is a network read performed only
after account authority and recipient policy have accepted the request, never
during command validation. Only ``http``/``https`` and ``data`` schemes are
accepted; redirects to other schemes are refused. Retrieval size is bounded at
read time (a download can lie about ``Content-Length``), the per-attachment
ceiling applies to decoded content, and the caller's URL is never echoed into
logs — only the derived filename is logged.

Network guard: the server process would otherwise be an arbitrary fetcher for
the MCP caller. By default every request in the redirect chain must resolve to
a public network address — loopback, private, link-local (cloud metadata),
and otherwise reserved ranges are refused, re-checked on every hop. Deployments
that deliberately serve attachments from a private network can opt out with
``MCP_EMAIL_SERVER_ALLOW_PRIVATE_ATTACHMENT_HOSTS=1``; see
``docs/security.md``.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import os
import re
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, unquote_to_bytes, urljoin, urlparse

import httpx

from mcp_email_server.application.limits import APPLICATION_LIMITS, validate_controlled_string
from mcp_email_server.log import logger

_URL_SCHEMES = frozenset({"http", "https"})
_DOWNLOAD_CHUNK_BYTES = 64 * 1_024
_HTTP_TIMEOUT_SECONDS = 30.0
_PRIVATE_HOSTS_ENV = "MCP_EMAIL_SERVER_ALLOW_PRIVATE_ATTACHMENT_HOSTS"

# Loose RFC 2397 shape: data:[<mediatype>][;base64],<payload>
_DATA_URL_PATTERN = re.compile(r"^data:(?P<mediatype>[^,]*)?,(?P<payload>.*)\Z", re.DOTALL)

_SAFE_FILENAME_SUBSTITUTIONS = str.maketrans({"/": "_", "\\": "_", "\x00": "_"})


class AttachmentUrlError(ValueError):
    """Raised when a URL attachment reference cannot be resolved."""


@dataclass(frozen=True)
class MaterializedAttachments:
    """Combined attachment paths plus cleanup for the URL temp directory."""

    paths: tuple[str, ...]
    cleanup: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> tuple[str, ...]:
        return self.paths

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self.cleanup is not None:
            self.cleanup.cleanup()


def _url_ceiling_bytes(url: str) -> int:
    """Return the string-size ceiling for one URL: data: URLs carry content inline."""
    if url.lower().startswith("data:"):
        return APPLICATION_LIMITS.data_url_bytes
    return APPLICATION_LIMITS.attachment_path_bytes


def validate_attachment_urls(urls: tuple[str, ...]) -> None:
    """Bound caller-supplied attachment URLs without any network or disk I/O.

    Syntactic only, mirroring ``_validate_attachment_paths``: command
    validation runs before account authority is known, so no URL is fetched
    and no response byte is read here.
    """

    if len(urls) > APPLICATION_LIMITS.attachments:
        raise ValueError(f"attachment_urls must contain at most {APPLICATION_LIMITS.attachments} values")
    if any(not isinstance(raw_url, str) for raw_url in urls):
        raise ValueError("attachment URLs must be strings")
    for raw_url in urls:
        ceiling = _url_ceiling_bytes(raw_url)
        validate_controlled_string(
            raw_url,
            field_name="attachment URL",
            maximum_bytes=ceiling,
        )
        scheme = urlparse(raw_url).scheme.lower()
        if scheme not in _URL_SCHEMES and scheme != "data":
            raise ValueError("attachment URLs must use http, https, or data schemes")


def _sanitize_filename(candidate: str) -> str:
    """Reduce a URL-derived filename to one safe path component."""

    cleaned = candidate.translate(_SAFE_FILENAME_SUBSTITUTIONS).strip().lstrip(".")
    return cleaned[:255] or "attachment"


def _filename_from_http_url(url: str) -> str:
    basename = unquote(Path(urlparse(url).path).name) if urlparse(url).path else ""
    return _sanitize_filename(basename)


def _filename_from_data_mediatype(data_url: str) -> str:
    match = _DATA_URL_PATTERN.match(data_url)
    mediatype = match.group("mediatype") if match else ""
    parameters = mediatype.split(";") if mediatype else []
    for parameter in parameters[1:]:
        name, separator, value = parameter.partition("=")
        if separator and name.strip().lower() in {"filename", "name"}:
            cleaned = _sanitize_filename(value.strip().strip('"'))
            if cleaned:
                return cleaned
    return "attachment"


def _decode_data_url(data_url: str) -> bytes:
    match = _DATA_URL_PATTERN.match(data_url)
    if match is None:
        raise AttachmentUrlError("data URL does not match the RFC 2397 shape")
    mediatype, payload = match.group("mediatype") or "", match.group("payload")
    try:
        if mediatype.casefold().endswith(";base64"):
            return base64.b64decode(payload, validate=True)
        return unquote_to_bytes(payload)
    except (binascii.Error, ValueError) as exc:
        raise AttachmentUrlError("data URL payload could not be decoded") from exc


def _unique_path(directory: Path, filename: str, seen: set[str]) -> Path:
    """Pick a non-colliding materialized filename inside a private temp directory."""

    candidate = directory / filename
    counter = 1
    while candidate.name in seen:
        candidate = directory / f"{counter}-{filename}"
        counter += 1
    seen.add(candidate.name)
    return candidate


def _private_hosts_allowed() -> bool:
    """Explicit opt-out for serving attachments from a private network."""
    return os.getenv(_PRIVATE_HOSTS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _guard_resolved_address(host: str, ip: str) -> None:
    """Refuse a resolution result pointing at a non-public network address.

    By default the attachment fetcher must stay a public-network client: loopback,
    RFC 1918, link-local (including the 169.254/16 cloud metadata endpoint),
    and every other reserved range are refused. IPv6-mapped IPv4 addresses are
    unwrapped so the IPv4 classification still applies.
    """

    if _private_hosts_allowed():
        return
    address = ipaddress.ip_address(ip)
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if not address.is_global:
        raise AttachmentUrlError("attachment URL host resolves to a non-public network address")


def _host_addresses(hostname: str) -> list[str]:
    """Resolve the host to the addresses a connection could actually use."""

    try:
        infos = socket.getaddrinfo(hostname, 443, 0, socket.SOCK_STREAM)
    except OSError as exc:
        raise AttachmentUrlError("attachment URL host could not be resolved") from exc
    addresses = [str(info[4][0]).lstrip("[]") for info in infos]
    if not addresses:
        raise AttachmentUrlError("attachment URL host could not be resolved")
    return addresses


def assert_public_host(hostname: str) -> None:
    """Refuse a host that resolves entirely or partly into non-public address space.

    Called before every request hop (initial URL and each redirect), so a
    redirect to an internal destination is refused as well. The check runs at
    request time; DNS may change between the check and the connect, and the
    deployment-facing statement of that residual is in ``docs/security.md``.
    """

    if not hostname:
        raise AttachmentUrlError("attachment URL has no host")
    if _private_hosts_allowed():
        return
    for address in _host_addresses(hostname):
        _guard_resolved_address(hostname, address)


_MAX_REDIRECTS = 10
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})


def _check_content_ceiling(content: bytes) -> None:
    """Bound decoded content; both resolvers already enforce this at read time.

    Retained as the defensive net if either resolver path ever loses its bound.
    """

    if len(content) > APPLICATION_LIMITS.attachment_bytes:
        raise ValueError(f"an attachment exceeds {APPLICATION_LIMITS.attachment_bytes} bytes")


async def _fetch_http_url(url: str) -> bytes:
    """Download at most one byte over the per-attachment ceiling.

    ``Content-Length`` is untrusted, so the bound is enforced on streamed bytes.
    Redirects are followed manually (at most ``_MAX_REDIRECTS`` hops) so that
    each hop — not only the final URL — is re-checked: the scheme must stay
    http(s) and ``assert_public_host`` re-resolves the hop's host, refusing
    non-public address space on every hop.
    """

    ceiling = APPLICATION_LIMITS.attachment_bytes
    async with httpx.AsyncClient(follow_redirects=False, timeout=_HTTP_TIMEOUT_SECONDS) as client:
        current_url = url
        for _hop in range(_MAX_REDIRECTS + 1):
            parsed = urlparse(current_url)
            if parsed.scheme.lower() not in _URL_SCHEMES:
                raise AttachmentUrlError("attachment URL redirected to a non-http(s) scheme")
            assert_public_host(parsed.hostname or "")
            async with client.stream("GET", current_url) as response:
                if response.status_code in _REDIRECT_STATUS_CODES:
                    location = response.headers.get("location")
                    if not location:
                        raise AttachmentUrlError("attachment URL redirect is missing a Location header")
                    current_url = urljoin(current_url, location)
                    continue
                if response.status_code >= 400:
                    raise AttachmentUrlError(f"attachment URL request failed with HTTP {response.status_code}")
                chunks: list[bytes] = []
                downloaded = 0
                async for chunk in response.aiter_bytes(_DOWNLOAD_CHUNK_BYTES):
                    downloaded += len(chunk)
                    if downloaded > ceiling:
                        raise ValueError(f"an attachment exceeds {ceiling} bytes")
                    chunks.append(chunk)
                return b"".join(chunks)
    raise AttachmentUrlError("attachment URL exceeded the maximum number of redirects")


async def resolve_attachment_urls(attachments: tuple[str, ...], urls: tuple[str, ...]) -> MaterializedAttachments:
    """Validate, fetch, and materialize URL attachments; return combined paths.

    Returns the combined attachment list (local paths first, then materialized
    files) together with the temporary directory to clean up after the provider
    effect completes. Fetches happen here, which callers invoke only after the
    request has been authorized — mirroring when ``_preflight_attachment_sizes``
    is allowed to stat caller paths.
    """

    validate_attachment_urls(urls)
    if len(attachments) + len(urls) > APPLICATION_LIMITS.attachments:
        raise ValueError(f"attachments must contain at most {APPLICATION_LIMITS.attachments} paths")
    if not urls:
        return MaterializedAttachments(paths=(*attachments,))

    cleanup = tempfile.TemporaryDirectory(prefix="mcp-email-url-attachments-")
    directory = Path(cleanup.name)
    try:
        seen: set[str] = set()
        materialized: list[str] = []
        for url in urls:
            if url.lower().startswith("data:"):
                content = _decode_data_url(url)
            else:
                content = await _fetch_http_url(url)
            _check_content_ceiling(content)
            filename = _filename_from_data_mediatype(url) if url.lower().startswith("data:") else _filename_from_http_url(url)
            path = _unique_path(directory, filename, seen)
            path.write_bytes(content)
            materialized.append(str(path))
            logger.info(f"Materialized URL attachment: {path.name}")
        return MaterializedAttachments(paths=(*attachments, *materialized), cleanup=cleanup)
    except Exception:
        cleanup.cleanup()
        raise
