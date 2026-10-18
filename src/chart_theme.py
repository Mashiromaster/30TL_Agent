"""Shared Plotly styling for the research workspace.

One measure gets one hue. Category breakdowns are identity, not magnitude, so
bars never carry a value ramp. Colours are validated against the light chart
surface (#ffffff) for the lightness band, chroma floor and 3:1 contrast.
"""
from __future__ import annotations

SURFACE = "#ffffff"
INK = "#173044"
INK_MUTED = "#526578"
GRID = "#dce4e9"

SERIES = "#2a78d6"          # categorical slot 1 — the default single hue
SERIES_ALT = "#1baf7a"      # slot 2, only when two measures share one plot
REFERENCE = "#9aa9b5"       # neutral baseline / annotation rule

FONT = dict(family="-apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif",
            size=12, color=INK)


def apply_theme(fig, *, height=300, showlegend=False, x_title=None, y_title=None):
    """Recessive chrome, hairline solid grid, generous padding."""
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        margin=dict(l=8, r=16, t=8, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=FONT,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=FONT),
        bargap=0.45,
    )
    fig.update_xaxes(title_text=x_title, showgrid=False, zeroline=False,
                     linecolor=GRID, linewidth=1, ticks="outside",
                     tickcolor=GRID, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, gridwidth=1,
                     griddash="solid", zeroline=False, showline=False,
                     tickfont=dict(color=INK_MUTED))
    return fig


def reference_line(fig, value, label=None, *, orientation="h"):
    """A solid hairline, never a dashed pseudo-threshold.

    ``label=None`` leaves the rule unannotated — use it when marks occupy the
    plot edges and an inline label would collide; carry the meaning in a caption.
    """
    kwargs = dict(line_color=REFERENCE, line_width=1)
    # Passing any annotation_* key makes Plotly add an annotation, so only
    # supply them when there is a label to show.
    if label:
        kwargs.update(annotation_text=label,
                      annotation_font=dict(color=INK_MUTED, size=11),
                      annotation_position="top left" if orientation == "h" else "top")
    if orientation == "h":
        fig.add_hline(y=value, **kwargs)
    else:
        fig.add_vline(x=value, **kwargs)
    return fig


def bar_width(count, *, plot_px=440, max_px=24):
    """Category-axis width that caps rendered bar thickness at ``max_px``.

    One category spans ``plot_px / count`` pixels, so ``max_px`` pixels is that
    fraction of a band. Capped at 0.6 of a band so densely packed categories
    keep a visible gap; below that the cap is exact.
    """
    if count <= 0:
        return None
    band_px = plot_px / count
    return min(0.6, max_px / band_px)
