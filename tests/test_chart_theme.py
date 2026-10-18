"""Chart rules that a screenshot caught but a figure spec did not."""
import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chart_theme


def spec(fig):
    return json.loads(fig.to_json())


def test_unlabelled_reference_line_adds_no_annotation():
    # Passing annotation_font without annotation_text makes Plotly inject a
    # default "new text" annotation into the plot.
    fig = chart_theme.reference_line(go.Figure(go.Bar(x=["a"], y=[1])), 50)
    layout = spec(fig)["layout"]
    assert layout.get("annotations", []) == []
    assert [s["line"]["dash"] for s in layout["shapes"] if "dash" in s["line"]] == []


def test_labelled_reference_line_keeps_its_text():
    fig = chart_theme.reference_line(go.Figure(go.Bar(x=["a"], y=[1])), 50, "随机水平 50%")
    assert [a["text"] for a in spec(fig)["layout"]["annotations"]] == ["随机水平 50%"]


@pytest.mark.parametrize("count", [1, 2, 3, 6, 12, 30])
def test_bar_width_caps_rendered_thickness(count):
    plot_px = 440
    width = chart_theme.bar_width(count, plot_px=plot_px, max_px=24)
    # Never thicker than the 24px mark spec, and never a full band (the
    # leftover band space is the gap between neighbours).
    assert width * plot_px / count <= 24 + 1e-9
    assert 0 < width <= 0.6


def test_bar_width_rejects_empty_category_set():
    assert chart_theme.bar_width(0) is None


def test_theme_grid_is_solid_hairline_on_the_light_surface():
    fig = chart_theme.apply_theme(go.Figure(go.Bar(x=["a"], y=[1])))
    layout = spec(fig)["layout"]
    assert layout["plot_bgcolor"] == chart_theme.SURFACE
    assert layout["yaxis"]["griddash"] == "solid"
    assert layout["yaxis"]["gridwidth"] == 1
    assert layout["xaxis"]["showgrid"] is False
