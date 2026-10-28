from __future__ import annotations

import locale

import matplotlib as mpl
from cycler import cycler


PALETTE = {
    "azul": "#1F5FA8",
    "rojo": "#C62828",
    "naranja": "#C96A00",
    "verde_azulado": "#138A72",
    "gris": "#A3A3A3",
    "verde_agp": "#3B8E4F",
    "ambar": "#B7791F",
}

RAMPS = {
    "azul": {"oscuro": "#0C325C", "base": "#1F5FA8", "claro": "#84A6D1", "muy_claro": "#D1DEEF"},
    "rojo": {"oscuro": "#6E1212", "base": "#C62828", "claro": "#E89288", "relleno": "#F6C4BD",
             "muy_claro": "#FAD7D3"},
    "naranja": {"oscuro": "#6F3800", "base": "#C96A00", "claro": "#E5AE87", "muy_claro": "#F7E1D3"},
    "verde_azulado": {"oscuro": "#064B3D", "base": "#138A72", "claro": "#8CBEAF",
                      "muy_claro": "#D5E7E1"},
    "gris": {"oscuro": "#595959", "base": "#A3A3A3", "claro": "#CBCBCB", "muy_claro": "#ECECEC"},
    "verde_agp": {"base": "#3B8E4F", "muy_claro": "#D8E8DA"},
}

#: Neutral ink for text, axes and the measured signal. Text never takes a series colour.
INK = {
    "texto": "#262626",
    "secundario": "#5C5C5C",
    "tenue": "#8A8A8A",
    "rejilla": "#E6E6E6",
    "fondo": "#FFFFFF",
}

#: What each colour *means* across the memoir. Notebooks use these names.
ROLE = {
    # The proposal: the model, its predictions, HUPA as the cohort under study.
    "modelo": PALETTE["azul"],
    "modelo_banda": RAMPS["azul"]["muy_claro"],
    # What the proposal is compared against: baselines, OhioT1DM as reference.
    "referencia": PALETTE["gris"],
    "referencia_banda": RAMPS["gris"]["muy_claro"],
    # Hypoglycaemia and everything that warns of it: < 70 line, events, alarms.
    # Reserved: never a plain category.
    "hipo": PALETTE["rojo"],
    "hipo_relleno": RAMPS["rojo"]["relleno"],
    # Clinical bands of a glucose axis, as in the AGP report.
    "rango_objetivo": RAMPS["verde_agp"]["muy_claro"],
    "hiper": PALETTE["ambar"],
    # The measured glucose is data, not a model: neutral ink.
    "observado": INK["texto"],
    "destacado": RAMPS["azul"]["oscuro"],
}

#: Categorical order for identity, assigned in sequence and never cycled. Three
#: colours, all pairs validated; a fourth series is the reference grey
#: (``ROLE["referencia"]``) or goes to facets. Orange is skipped in a figure that
#: shows the hypoglycaemia red.
CATEGORICAL = (PALETTE["azul"], PALETTE["naranja"], PALETTE["verde_azulado"])

#: Magnitude: one hue, light to dark.
SEQUENTIAL = ("#E4EBF5", "#B0C6E2", "#7EA2CF", "#4E7FBB", "#1F5FA8", "#0C325C")

#: Polarity around a baseline: red (worse / below) - neutral - blue (better / above).
DIVERGING = (RAMPS["rojo"]["base"], RAMPS["rojo"]["claro"], "#F2F2F2",
             RAMPS["azul"]["claro"], RAMPS["azul"]["base"])

#: Sans-serif for figures, first available wins. Arial exists on any machine the
#: delivered code is likely to run on; DejaVu Sans ships with matplotlib.
FONT_FAMILY = ["Arial", "Helvetica", "DejaVu Sans"]

#: Widths that fit the memoir's text block (A4, 2.5 cm margins: 16 cm).
FIG_WIDTH_FULL = 6.3   # inches, full text width
FIG_WIDTH_HALF = 3.1   # inches, one of two side by side


def sequential_cmap():
    """Matplotlib colormap of :data:`SEQUENTIAL`, for heatmaps and gradients."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("clinica_secuencial", SEQUENTIAL)


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("clinica_divergente", DIVERGING)


def _spanish_decimal_comma() -> bool:
    """Decimal comma on tick labels, as Spanish typesetting requires.

    Only ``LC_NUMERIC`` is touched, which affects number *formatting*; Python and
    pandas parse numbers with their own rules. Silently skipped if the locale is
    not installed, so the figures still render elsewhere.
    """
    for name in ("es_ES.UTF-8", "es_ES.utf8", "es_ES"):
        try:
            locale.setlocale(locale.LC_NUMERIC, name)
            return True
        except locale.Error:
            continue
    return False


def apply() -> None:
    """Set matplotlib's defaults to the TFM style. Called by ``set_style``."""
    comma = _spanish_decimal_comma()
    mpl.rcParams.update({
        # Canvas and export
        "figure.dpi": 110,
        "figure.facecolor": INK["fondo"],
        "figure.figsize": (FIG_WIDTH_FULL, 2.6),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.facecolor": INK["fondo"],
        "pdf.fonttype": 42,          # embed TrueType: selectable text in the PDF
        "ps.fonttype": 42,
        # Type
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "text.color": INK["texto"],
        "axes.labelcolor": INK["texto"],
        "axes.titlecolor": INK["texto"],
        "xtick.color": INK["secundario"],
        "ytick.color": INK["secundario"],
        "axes.formatter.use_locale": comma,
        "axes.unicode_minus": True,
        # Axes: recessive frame and grid, so the data carries the ink
        "axes.facecolor": INK["fondo"],
        "axes.edgecolor": INK["tenue"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": INK["rejilla"],
        "grid.linewidth": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "xtick.major.width": 0.8,
        # Marks
        "axes.prop_cycle": cycler(color=list(CATEGORICAL)),
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "patch.linewidth": 0,
        "hist.bins": 40,
        "errorbar.capsize": 2.5,
        "scatter.edgecolors": "face",
        "boxplot.medianprops.color": INK["texto"],
        # Legend: no box, it sits on the data's white space
        "legend.frameon": False,
        "legend.borderaxespad": 0.3,
        "legend.handlelength": 1.6,
        "legend.labelspacing": 0.35,
        "image.cmap": "viridis",
    })
    mpl.colormaps.register(sequential_cmap(), force=True)
    mpl.colormaps.register(diverging_cmap(), force=True)
    mpl.rcParams["image.cmap"] = "clinica_secuencial"


def fmt_es(x: float, decimals: int = 2) -> str:
    """A number with a decimal comma, for titles and annotations."""
    return f"{x:.{decimals}f}".replace(".", ",")
