import json
import re
from pathlib import Path

import pytest

from tfm.paths import ROOT
from tfm.viz import style

HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
NAMED = re.compile(r"""color\s*=\s*["'](?:red|blue|green|orange|purple|grey|gray|black|k|tab:\w+)["']""")
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


def _code(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_ask_for_colours_by_role(nb_path):
    code = _code(nb_path)
    assert not HEX.findall(code), f"{nb_path.name}: hex colour written by hand"
    assert not NAMED.findall(code), f"{nb_path.name}: named colour written by hand"


def test_plotting_helpers_take_colours_from_the_style_module():
    src = (ROOT / "src" / "tfm" / "viz" / "plots.py").read_text()
    assert not HEX.findall(src)


def test_every_role_is_a_palette_colour_or_a_derived_one():
    allowed = set(style.PALETTE.values()) | set(style.INK.values())
    for ramp in style.RAMPS.values():
        allowed |= set(ramp.values())
    for role, colour in style.ROLE.items():
        assert colour in allowed, f"ROLE[{role!r}] = {colour} is not in the palette"


def test_fills_are_never_line_roles():
    # Tints are below 3:1 on white: they may fill, never draw.
    line_roles = ("modelo", "referencia", "hipo", "observado", "hiper", "destacado")
    fills = {style.ROLE[r] for r in ("hipo_relleno", "rango_objetivo", "modelo_banda", "referencia_banda")}
    assert all(style.ROLE[r] not in fills for r in line_roles)
    assert not fills & set(style.CATEGORICAL)


def test_hypoglycaemia_red_is_reserved():
    # Red means hypoglycaemia in every figure: never a plain category.
    assert style.ROLE["hipo"] not in style.CATEGORICAL


def test_style_applies_cleanly():
    import matplotlib as mpl

    style.apply()
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"][0].lower() == style.ROLE["modelo"].lower()
    assert "clinica_secuencial" in mpl.colormaps
