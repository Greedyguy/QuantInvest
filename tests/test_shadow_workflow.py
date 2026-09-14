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


def test_eod_workflow_records_frozen_kodex200_shadow_without_orders():
    workflow = Path(".github/workflows/daily-eod-signal.yml").read_text(
        encoding="utf-8"
    )

    assert "scripts/report_k200_reentry_shadow.py" in workflow
    assert "--output-dir reports/signals" in workflow
