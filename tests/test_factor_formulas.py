"""The formula registry must stay tied to the code that computes the factors."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import factor_formulas as ff


def test_every_entry_is_self_consistent():
    for name, entry in ff.FORMULAS.items():
        assert entry.name == name
        assert entry.formula.strip()
        assert entry.inputs, name
        assert entry.window.strip(), name
        assert entry.normalization.strip(), name
        assert entry.source.strip(), name
        assert entry.status in (ff.LIVE, ff.RETIRED), name


def test_retired_factors_name_their_replacement_and_say_so():
    retired = [e for e in ff.FORMULAS.values() if e.status == ff.RETIRED]
    assert retired, "the registry documents retired columns; keep at least one"
    for entry in retired:
        assert entry.replacement, entry.name
        assert any("不再计算" in note for note in entry.notes), entry.name


@pytest.mark.parametrize("name,entry", [
    (n, e) for n, e in ff.FORMULAS.items() if e.status == ff.LIVE
])
def test_live_factor_source_citation_resolves_to_real_lines(name, entry):
    """A live factor's cited file:line range must exist and mention its name."""
    cited = re.findall(r"(src/[\w/]+\.py):(\d+)(?:-(\d+))?", entry.source)
    assert cited, f"{name} has no parsable src citation: {entry.source}"
    for path, start, end in cited:
        lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
        first, last = int(start), int(end or start)
        assert 1 <= first <= last <= len(lines), f"{name}: {path}:{start}-{end}"
        # Widened a little: an assignment may sit a line or two off the citation.
        window = "\n".join(lines[max(0, first - 3):min(len(lines), last + 2)])
        assert name in window, f"{name} not found near {path}:{start}"


def test_retired_factors_cite_a_source_but_are_not_line_checked():
    # Their code is gone from HEAD, so a line citation would be meaningless.
    for entry in ff.FORMULAS.values():
        if entry.status == ff.RETIRED:
            assert not re.search(r"\.py:\d", entry.source), entry.name


def test_lookup_returns_none_for_unregistered_names():
    assert ff.lookup("Not_A_Factor") is None
    assert ff.lookup(None) is None


def test_pipeline_note_states_the_zero_ambiguity():
    # Zero-filling makes 0 ambiguous; the UI must not imply 0 means "zero".
    assert "缺失" in ff.PIPELINE_NOTE and "0" in ff.PIPELINE_NOTE
