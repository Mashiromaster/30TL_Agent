"""Selected feature names must match what the pipeline actually computes.

A name that no code assigns is filtered out by the ``available`` guard in the
retrain scripts, so the model trains on a quietly shorter feature set and
nothing errors. These tests catch that drift at the name level.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import factor_formulas as ff
from feature_registry import MACRO_PATTERNS


# Names kept out of the drift check: they are produced by name-pattern loops or
# by modules outside the factor pipeline, not by a literal assignment.
GENERATED_PREFIXES = ("CNN_Emb_", "JEV_", "GP_")


def _assigning_sources():
    text = {}
    for path in SRC.glob("*.py"):
        if path.name in {"retrain_v2.py", "retrain_optimized.py", "factor_formulas.py"}:
            continue  # consumers and docs, not producers
        text[path.name] = path.read_text(encoding="utf-8", errors="replace")
    return text


def _is_computed(name, sources):
    if name.startswith(GENERATED_PREFIXES):
        return True
    pattern = re.compile(
        r"""\[['"]%s['"]\]\s*=|["']%s["']\s*:""" % (re.escape(name), re.escape(name)))
    return any(pattern.search(body) for body in sources.values())


def _selected_features():
    """Read the literal list out of retrain_v2 without running training."""
    body = (SRC / "retrain_v2.py").read_text(encoding="utf-8")
    block = body.split("selected_features = [", 1)[1].split("]", 1)[0]
    return re.findall(r"'([A-Za-z0-9_]+)'", block)


def test_selected_features_are_all_computed_somewhere():
    sources = _assigning_sources()
    missing = [n for n in _selected_features() if not _is_computed(n, sources)]
    assert not missing, (
        "these selected features are not computed by any module, so training "
        "would silently drop them: " + ", ".join(missing))


def test_selected_feature_list_has_no_duplicates():
    names = _selected_features()
    assert len(names) == len(set(names))


@pytest.mark.parametrize("replacement", sorted({
    e.replacement for e in ff.FORMULAS.values() if e.replacement}))
def test_each_documented_replacement_is_itself_registered(replacement):
    # A retired factor pointing at an undocumented replacement leaves the
    # reader with a name and no formula.
    assert replacement in ff.FORMULAS


def test_macro_pattern_table_matches_live_macro_factor_names():
    live_macro = [e.name for e in ff.FORMULAS.values()
                  if e.category == "宏观" and e.status == ff.LIVE]
    assert live_macro
    unmatched = [n for n in live_macro
                 if not any(p in n for p in MACRO_PATTERNS)]
    assert not unmatched, (
        "auto-detection by prefix would miss these macro factors: "
        + ", ".join(unmatched))


def test_retrain_scripts_do_not_reintroduce_retired_names():
    retired = [e.name for e in ff.FORMULAS.values() if e.status == ff.RETIRED]
    for script in ("retrain_v2.py", "retrain_optimized.py"):
        body = (SRC / script).read_text(encoding="utf-8")
        for name in retired:
            # A mention in a comment is fine; a quoted feature name is not.
            assert f"'{name}'" not in body and f'"{name}"' not in body, (
                f"{script} still selects retired factor {name}")
