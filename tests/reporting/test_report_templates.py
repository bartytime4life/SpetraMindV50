from spectramind.reporting.report_templates import get_registry


def test_template_registry_smoke(tmp_path):
    reg = get_registry()
    ctx = reg.minimal_context()

    templates = reg.list_templates()
    assert templates  # registry discovers templates

    # Ensure each template renders with minimal context
    for t in templates:
        rendered = reg.render_to_string(t.name, ctx)
        assert isinstance(rendered, str)
        if t.name != "macros.html.j2":
            assert rendered.strip()

    # Render one template to file and copy assets
    out_html = tmp_path / "report.html"
    reg.render_to_file(
        "diagnostic_report.html.j2", ctx, str(out_html), copy_assets=True
    )
    assert out_html.exists()
    assert (tmp_path / "assets").is_dir()

    # Inline asset helper
    css_text = reg.inline_asset("assets/spectramind_report.css")
    assert ".container" in css_text
