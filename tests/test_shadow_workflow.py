from pathlib import Path


def test_shadow_workflow_persists_report_without_pushing_to_main():
    workflow = Path(".github/workflows/daily-open-exec-b-shadow.yml").read_text(
        encoding="utf-8"
    )

    assert "actions/upload-artifact@v4" in workflow
    assert "retention-days: 90" in workflow
    assert "contents: read" in workflow
    assert "git push" not in workflow
    assert "git rebase" not in workflow
