"""Tests for URL-supplied attachments (http/https and data: base64)."""

import base64
from email import message_from_bytes
from pathlib import Path

import pytest

from mcp_email_server.application.limits import APPLICATION_LIMITS
from mcp_email_server.emails.attachment_urls import (
    AttachmentUrlError,
    resolve_attachment_urls,
    validate_attachment_urls,
)


class TestValidateAttachmentUrls:
    def test_accepts_http_https_and_data_schemes(self):
        validate_attachment_urls(
            (
                "https://example.com/file.pdf",
                "http://example.com/file.pdf",
                "data:application/pdf;base64,SGVsbG8=",
            )
        )

    def test_rejects_other_schemes(self):
        for url in (
            "file:///etc/passwd",
            "ftp://example.com/file.pdf",
            "gopher://example.com",
            "javascript:alert(1)",
        ):
            with pytest.raises(ValueError, match="http, https, or data"):
                validate_attachment_urls((url,))

    def test_rejects_too_many_urls(self):
        urls = tuple(f"https://example.com/file{i}.pdf" for i in range(APPLICATION_LIMITS.attachments + 1))
        with pytest.raises(ValueError, match="at most"):
            validate_attachment_urls(urls)

    def test_rejects_non_string_entries(self):
        with pytest.raises(ValueError, match="strings"):
            validate_attachment_urls((42,))  # type: ignore[arg-type]

    def test_rejects_oversized_url(self):
        with pytest.raises(ValueError):
            validate_attachment_urls(("https://example.com/" + "a" * APPLICATION_LIMITS.attachment_path_bytes,))


class TestResolveAttachmentUrls:
    async def test_data_url_base64_is_materialized(self, tmp_path):
        payload = base64.b64encode(b"PDF content here").decode()
        data_url = f"data:application/pdf;base64,{payload}"
        materialized = await resolve_attachment_urls((), (data_url,))
        try:
            assert len(materialized.paths) == 1
            materialized_path = Path(materialized.paths[0])
            assert materialized_path.read_bytes() == b"PDF content here"
        finally:
            materialized.close()

    async def test_data_url_with_filename_parameter(self):
        payload = base64.b64encode(b"data").decode()
        data_url = f"data:text/plain;name=notes.txt;base64,{payload}"
        materialized = await resolve_attachment_urls((), (data_url,))
        try:
            assert Path(materialized.paths[0]).name == "notes.txt"
        finally:
            materialized.close()

    async def test_http_url_is_downloaded(self, tmp_path, monkeypatch):
        from mcp_email_server.emails import attachment_urls

        class FakeResponse:
            status_code = 200
            url = "https://example.com/report.pdf"

            def __init__(self, content):
                self._content = content

            def aiter_bytes(self, _chunk):
                async def gen():
                    yield self._content

                return gen()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

        class FakeClient:
            def __init__(self, *_args, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            def stream(self, _method, url):
                return FakeResponse(b"%PDF-1.4 fake content")

        monkeypatch.setattr(attachment_urls.httpx, "AsyncClient", FakeClient)
        materialized = await resolve_attachment_urls((), ("https://example.com/report.pdf",))
        try:
            assert len(materialized.paths) == 1
            assert Path(materialized.paths[0]).name == "report.pdf"
            assert Path(materialized.paths[0]).read_bytes() == b"%PDF-1.4 fake content"
        finally:
            materialized.close()

    async def test_local_paths_pass_through(self, tmp_path):
        local = tmp_path / "local.txt"
        local.write_bytes(b"local")
        data_url = "data:text/plain;base64," + base64.b64encode(b"remote").decode()
        materialized = await resolve_attachment_urls((str(local),), (data_url,))
        try:
            assert len(materialized.paths) == 2
            assert materialized.paths[0] == str(local)
            assert Path(materialized.paths[1]).read_bytes() == b"remote"
        finally:
            materialized.close()

    async def test_combined_budget_is_enforced(self, tmp_path):
        local = tmp_path / "local.txt"
        local.write_bytes(b"local")
        urls = tuple(f"https://example.com/file{i}.pdf" for i in range(APPLICATION_LIMITS.attachments))
        with pytest.raises(ValueError, match="at most"):
            await resolve_attachment_urls((str(local),), urls)

    async def test_http_error_raises(self, monkeypatch):
        from mcp_email_server.emails import attachment_urls

        class FakeResponse:
            status_code = 404
            url = "https://example.com/missing.pdf"

            def aiter_bytes(self, _chunk):
                async def gen():
                    return
                    yield

                return gen()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

        class FakeClient:
            def __init__(self, *_args, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            def stream(self, _method, _url):
                return FakeResponse()

        monkeypatch.setattr(attachment_urls.httpx, "AsyncClient", FakeClient)
        with pytest.raises(AttachmentUrlError, match="HTTP 404"):
            await resolve_attachment_urls((), ("https://example.com/missing.pdf",))

    async def test_invalid_base64_raises(self):
        with pytest.raises(AttachmentUrlError, match="decoded"):
            await resolve_attachment_urls((), ("data:text/plain;base64,!!!not-base64!!!",))

    async def test_empty_urls_returns_paths_without_cleanup(self):
        result = await resolve_attachment_urls(("a", "b"), ())
        assert result.paths == ("a", "b")
        assert result.cleanup is None
        result.close()  # no-op

    async def test_temp_directory_is_removed_after_close(self):
        data_url = "data:text/plain;base64," + base64.b64encode(b"x").decode()
        result = await resolve_attachment_urls((), (data_url,))
        directory = Path(result.paths[0]).parent
        assert directory.exists()
        result.close()
        assert not directory.exists()


class TestComposeCommandAttachmentUrls:
    def test_compose_command_accepts_attachment_urls(self):
        from mcp_email_server.application.mutations import SendCommand

        command = SendCommand(
            account_name="test",
            recipients=("to@example.com",),
            subject="subject",
            body="body",
            attachment_urls=("https://example.com/file.pdf",),
        )
        command.validate()

    def test_compose_command_rejects_bad_scheme(self):
        from mcp_email_server.application.mutations import SendCommand

        command = SendCommand(
            account_name="test",
            recipients=("to@example.com",),
            subject="subject",
            body="body",
            attachment_urls=("file:///etc/passwd",),
        )
        with pytest.raises(ValueError, match="http, https, or data"):
            command.validate()


class TestSendEmailWithUrlAttachments:
    async def test_send_email_with_data_url_attachment_end_to_end(self, email_client_fixture, tmp_path):
        """A data: URL attachment reaches the composed MIME message as a real part."""
        payload = base64.b64encode(b"e2e attachment payload").decode()
        data_url = f"data:application/pdf;base64,{payload}"
        materialized = await resolve_attachment_urls((), (data_url,))
        message = email_client_fixture.compose_message(
            recipients=["to@example.com"],
            subject="url attachment",
            body="see attached",
            attachments=list(materialized.paths),
        )
        materialized.close()
        raw = message.as_bytes()
        parsed = message_from_bytes(raw)
        parts = [p for p in parsed.walk() if p.get_filename()]
        assert len(parts) == 1
        assert parts[0].get_payload(decode=True) == b"e2e attachment payload"


@pytest.fixture
def email_client_fixture():
    from mcp_email_server.config import EmailServer
    from mcp_email_server.emails.classic import EmailClient

    server = EmailServer(
        user_name="test_user",
        password="test_password",
        host="smtp.example.com",
        port=465,
        use_ssl=True,
    )
    return EmailClient(server, sender="Test User <test@example.com>")
