from askbook.observability.redact import Redactor


def test_redact_email() -> None:
    r = Redactor()
    result = r.apply("联系 alice@example.com 获取详情")
    assert result == "联系 <REDACTED:EMAIL> 获取详情"


def test_redact_cn_phone() -> None:
    r = Redactor()
    result = r.apply("电话 13800138000 联系我")
    assert "<REDACTED:PHONE>" in result


def test_redact_jwt_like_token() -> None:
    r = Redactor()
    # Any string of 32+ alphanumeric chars (like a JWT token) should be redacted
    result = r.apply("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abc1234567890")
    assert "<REDACTED:TOKEN>" in result


def test_redact_passthrough_when_disabled() -> None:
    r = Redactor(enabled=False)
    text = "alice@example.com 13800138000"
    assert r.apply(text) == text


def test_redact_recursive_on_dict() -> None:
    r = Redactor()
    data = {"q": "alice@x.com", "nested": {"phone": "13800138000"}}
    result = r.apply_to_mapping(data)
    assert result["q"] == "<REDACTED:EMAIL>"
    assert result["nested"]["phone"] == "<REDACTED:PHONE>"
