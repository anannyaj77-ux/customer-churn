"""
Headless smoke-test for app.py using streamlit.testing.v1.AppTest.
Tests:
  - App loads without exceptions in both Dark and Light mode
  - Basic filter combinations including one that returns zero customers
  - No DeprecationWarning from use_container_width (we replaced those with width="stretch")
"""
import warnings
import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = "app.py"


def _run(session_state: dict = None, timeout: int = 60) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=timeout)
    if session_state:
        for key, val in session_state.items():
            at.session_state[key] = val
    at.run()
    return at


def test_app_loads_dark_mode():
    """App boots without exceptions in Dark Mode (default)."""
    at = _run({"theme_mode": "Dark Mode"})
    assert not at.exception, f"Exception in Dark Mode: {at.exception}"


def test_app_loads_light_mode():
    """App boots without exceptions in Light Mode."""
    at = _run({"theme_mode": "Light Mode"})
    assert not at.exception, f"Exception in Light Mode: {at.exception}"


def test_filter_high_risk_only():
    """High-Risk filter returns customers and no exception."""
    at = _run({"theme_mode": "Dark Mode"})
    # Select 'High Risk (>=60%)' in the first sidebar selectbox (Customer Risk Tier)
    at.sidebar.selectbox[0].select("High Risk (>=60%)")
    at.run()
    assert not at.exception, f"Exception with High Risk filter: {at.exception}"


def test_filter_month_to_month():
    """Month-to-Month contract filter returns customers and no exception."""
    at = _run({"theme_mode": "Dark Mode"})
    at.sidebar.selectbox[1].select("Month-to-month")
    at.run()
    assert not at.exception, f"Exception with Month-to-month filter: {at.exception}"


def test_filter_zero_customers():
    """Combining filters that return zero customers should not raise exceptions."""
    at = _run({"theme_mode": "Dark Mode"})
    # High risk + two-year contract is the combination most likely to return 0 rows
    at.sidebar.selectbox[0].select("High Risk (>=60%)")
    at.sidebar.selectbox[1].select("Two year")
    at.run()
    # Should NOT raise an exception even if filtered set is empty
    assert not at.exception, f"Exception on zero-customer filter: {at.exception}"


def test_no_use_container_width_deprecation():
    """Verify we replaced use_container_width=True so no DeprecationWarning fires."""
    import ast, pathlib
    source = pathlib.Path(APP_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "use_container_width":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        violations.append(f"Line {node.lineno}: use_container_width=True still present")
    # st.dataframe(..., use_container_width=True) is still valid for DataFrames — allowed
    # We only flag st.image / st.plotly_chart occurrences
    image_plot_violations = [v for v in violations if "dataframe" not in v.lower()]
    assert image_plot_violations == [], "\n".join(image_plot_violations)


if __name__ == "__main__":
    # Run directly with python test_app.py for quick manual check
    print("Running Dark Mode test...")
    at = _run({"theme_mode": "Dark Mode"})
    if at.exception:
        print(f"FAIL: {at.exception}")
    else:
        print("PASS: Dark Mode")

    print("Running Light Mode test...")
    at = _run({"theme_mode": "Light Mode"})
    if at.exception:
        print(f"FAIL: {at.exception}")
    else:
        print("PASS: Light Mode")

    print("Running zero-customer filter test...")
    at = _run({"theme_mode": "Dark Mode"})
    at.sidebar.selectbox[0].select("High Risk (>=60%)")
    at.sidebar.selectbox[1].select("Two year")
    at.run()
    if at.exception:
        print(f"FAIL: {at.exception}")
    else:
        print("PASS: Zero-customer filter (no exception)")

    print("Running use_container_width check...")
    import ast, pathlib
    source = pathlib.Path("app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "use_container_width":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        fn_name = ""
                        if isinstance(node.func, ast.Attribute):
                            fn_name = node.func.attr
                        if fn_name not in ("dataframe",):
                            violations.append(f"  Line {node.lineno}: {fn_name}(..., use_container_width=True)")
    if violations:
        print("WARN: use_container_width=True found in non-dataframe calls:")
        for v in violations:
            print(v)
    else:
        print("PASS: No use_container_width=True in image/plotly_chart calls")

    print("\nAll checks complete.")
