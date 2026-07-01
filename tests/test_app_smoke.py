import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_no_regular_button_is_nested_inside_streamlit_form():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))

    def is_streamlit_call(node: ast.AST, method: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
            and node.func.attr == method
        )

    violations: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        if not any(is_streamlit_call(item.context_expr, "form") for item in node.items):
            continue
        for child in ast.walk(node):
            if is_streamlit_call(child, "button"):
                violations.append(child.lineno)

    assert not violations, f"st.button() cannot be used inside st.form(); lines: {violations}"


def test_streamlit_app_starts_without_exceptions():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert not any("Missing Submit Button" in warning.value for warning in app.warning)


def test_measured_biomarker_profile_runs_end_to_end():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    app.selectbox[0].select("Measured Adult Research Profile")
    next(button for button in app.button if button.label == "Load Profile").click()
    app.run(timeout=30)

    next(button for button in app.button if button.label == "Run Simulation").click()
    app.run(timeout=60)

    assert not app.exception
    evidence_metric = next(metric for metric in app.metric if metric.label == "Input Evidence Score")
    assert evidence_metric.value == "80/100"
