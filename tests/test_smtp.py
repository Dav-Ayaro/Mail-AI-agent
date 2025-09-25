import pytest
from smtp_utils import send_email

def test_send_email(monkeypatch):
    def fake_sendmail(from_addr, to_addrs, msg):
        assert from_addr
        assert to_addrs
        assert "Test Body" in msg

    class FakeSMTP:
        def __init__(self, host, port): pass
        def starttls(self): pass
        def login(self, user, pwd): pass
        def sendmail(self, from_addr, to_addrs, msg): fake_sendmail(from_addr, to_addrs, msg)
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass

    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    send_email("to@example.com", "Test Subject", "Test Body")
