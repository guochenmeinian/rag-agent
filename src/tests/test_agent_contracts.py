from agent.contracts import RewriteResult


class TestRewriteResult:
    def test_parse_valid_payload(self):
        result = RewriteResult.parse({"type": "rewrite", "content": "ET5 续航是多少？"}, fallback="raw")
        assert result.type == "rewrite"
        assert result.content == "ET5 续航是多少？"

    def test_parse_invalid_payload_falls_back(self):
        result = RewriteResult.parse({"kind": "rewrite"}, fallback="原始问题")
        assert result.type == "rewrite"
        assert result.content == "原始问题"

    def test_parse_json_string(self):
        result = RewriteResult.parse('{"type":"clarify","content":"您是指 ET5 还是 ET7？"}', fallback="raw")
        assert result.type == "clarify"
        assert "ET5" in result.content
