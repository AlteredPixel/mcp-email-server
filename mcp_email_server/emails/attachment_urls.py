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
"""

from __future__ import annotations

import base64
import binascii
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from mcp_email_server.application.limits import APPLICATION_LIMITS, validate_controlled_string
from mcp_email_server.log import logger

_URL_SCHEMES = frozenset({"http", "https"})
_DOWNLOAD_CHUNK_BYTES = 64 * 1_024
_HTTP_TIMEOUT_SECONDS = 30.0

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


def is_attachment_url(reference: str) -> bool:
    """Return True when the reference is a URL form this module resolves."""

    scheme = urlparse(reference).scheme.lower()
    return scheme in _URL_SCHEMES or scheme == "data"


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
        validate_controlled_string(
            raw_url,
            field_name="attachment URL",
            maximum_bytes=APPLICATION_LIMITS.attachment_path_bytes,
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
        from urllib.parse import unquote_to_bytes

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


async def _fetch_http_url(url: str) -> bytes:
    """Download at most one byte over the per-attachment ceiling.

    ``Content-Length`` is untrusted, so the bound is enforced on streamed bytes.
    Redirects are followed by httpx but the final URL scheme is re-checked so a
    redirect can not escape to a non-http(s) scheme.
    """

    ceiling = APPLICATION_LIMITS.attachment_bytes
    async with httpx.AsyncClient(follow_redirects=True, timeout=_HTTP_TIMEOUT_SECONDS) as client:
        async with client.stream("GET", url) as response:
            if response.status_code >= 400:
                raise AttachmentUrlError(f"attachment URL request failed with HTTP {response.status_code}")
            chunks: list[bytes] = []
            downloaded = 0
            async for chunk in response.aiter_bytes(_DOWNLOAD_CHUNK_BYTES):
                downloaded += len(chunk)
                if downloaded > ceiling:
                    raise ValueError(f"an attachment exceeds {ceiling} bytes")
                chunks.append(chunk)
        _final_url = str(response.url)
    if urlparse(_final_url).scheme.lower() not in _URL_SCHEMES:
        raise AttachmentUrlError("attachment URL redirected to a non-http(s) scheme")
    return b"".join(chunks)


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
            if len(content) > APPLICATION_LIMITS.attachment_bytes:
                raise ValueError(f"an attachment exceeds {APPLICATION_LIMITS.attachment_bytes} bytes")
            filename = _filename_from_data_mediatype(url) if url.lower().startswith("data:") else _filename_from_http_url(url)
            path = _unique_path(directory, filename, seen)
            path.write_bytes(content)
            materialized.append(str(path))
            logger.info(f"Materialized URL attachment: {path.name}")
        return MaterializedAttachments(paths=(*attachments, *materialized), cleanup=cleanup)
    except Exception:
        cleanup.cleanup()
        raise
