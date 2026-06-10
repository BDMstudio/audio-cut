#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_module_contract_audit.py
# AI-SUMMARY: Tests auditing for new module file headers and public API docstrings.

from pathlib import Path

from scripts.audit_new_module_contracts import audit_paths


def test_audit_paths_reports_missing_header_and_public_docstring(tmp_path: Path) -> None:
    module_path = tmp_path / "bad_module.py"
    module_path.write_text("def public_function():\n    return None\n", encoding="utf-8")

    violations = audit_paths([module_path], require_public_docstrings=True)

    messages = [violation.message for violation in violations]
    assert "missing # File: header" in messages
    assert "missing # AI-SUMMARY: header" in messages
    assert "public function public_function missing docstring" in messages


def test_audit_paths_accepts_documented_module(tmp_path: Path) -> None:
    module_path = tmp_path / "good_module.py"
    module_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "# -*- coding: utf-8 -*-",
                "# File: good_module.py",
                "# AI-SUMMARY: Good module for audit tests.",
                "",
                "class PublicClass:",
                "    \"\"\"Documented public class.\"\"\"",
                "",
                "    def method(self):",
                "        return None",
                "",
                "def public_function():",
                "    \"\"\"Documented public function.\"\"\"",
                "    return None",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert audit_paths([module_path], require_public_docstrings=True) == []
