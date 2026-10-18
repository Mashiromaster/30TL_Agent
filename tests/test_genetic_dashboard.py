from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import genetic_dashboard


def fixture_frame():
    return pd.DataFrame({"date": pd.date_range("2026-01-01", periods=4), "ticker": "TL2609", "adj_close": [100., 101., 102., 103.], "signal": [1., 2., 3., 4.]})


def result(programs=True):
    return SimpleNamespace(
        miner=object(), factors=pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "ticker": ["TL2609"], "partition": ["validation"], "GP_000": [1.]}),
        report={
            "timestamp": "2026-09-30T18:00:00+08:00", "selection_basis": "validation_only",
            "holdout_evaluated": False, "n_final": int(programs),
            "mining_config": {"population_size": 1500, "generations": 15, "min_ic_threshold": .01, "max_new_factors": 20, "n_jobs": 1},
            "programs": [{"name": "GP_000", "expression": "add(signal, signal)", "validation_ic": .12, "complexity": 19, "candidate_index": 2}] if programs else [],
            "partitions": {"train": {"rows": 1000, "start": "2025-01-01", "end": "2025-06-30"}, "validation": {"rows": 500, "start": "2025-07-01", "end": "2025-09-30"}, "sealed_holdout": {"rows": 500, "start": "2025-10-01", "end": "2025-12-31"}},
        },
    )


def app_script():
    import os
    import pandas as pd
    from types import SimpleNamespace
    from genetic_dashboard import render_genetic_mining
    render_genetic_mining(SimpleNamespace(base_dir=os.environ["TEST_GP_BASE"], df_factors=pd.read_pickle(os.environ["TEST_GP_FRAME"])))


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = tmp_path / "fixture.pkl"
    fixture_frame().to_pickle(path)
    monkeypatch.setenv("TEST_GP_BASE", str(tmp_path))
    monkeypatch.setenv("TEST_GP_FRAME", str(path))
    return AppTest.from_function(app_script)


def test_new_miner_report_displays_and_survives_rerun(app, monkeypatch):
    mine = Mock(return_value=result())
    monkeypatch.setattr(genetic_dashboard, "run_discovery", mine)
    app.run()
    mine.assert_not_called()
    app.button(key="gp_run").click().run()
    assert not app.exception
    assert any("1" in message.value for message in app.success)
    assert any("验证" in str(table.value.columns) and "GP_000" in table.value.to_string() for table in app.dataframe)
    app.slider(key="gp_minic").set_value(.02).run()
    assert not app.exception
    mine.assert_called_once()
    assert app.button(key="gp_save_df")
    assert any("本次实际参数" in message.value and "最低 |IC| 0.01" in message.value for message in app.caption)


def test_save_result_is_reachable_without_mining_again(app, monkeypatch, tmp_path):
    discovered = result()
    mine = Mock(return_value=discovered)
    save = Mock(return_value=tmp_path / "outputs" / "genetic_factors" / "run")
    monkeypatch.setattr(genetic_dashboard, "run_discovery", mine)
    monkeypatch.setattr(genetic_dashboard, "save_discovery", save)
    app.run().button(key="gp_run").click().run()
    save.assert_not_called()
    app.button(key="gp_save_df").click().run()
    assert not app.exception
    mine.assert_called_once()
    save.assert_called_once_with(discovered, str(tmp_path))
    assert any("已保存" in message.value for message in app.success)
    assert not any("自动参与" in message.value for message in app.caption)


def test_failed_second_run_does_not_leave_stale_success(app, monkeypatch):
    mine = Mock(side_effect=[result(), ValueError("need at least 240 trading days")])
    monkeypatch.setattr(genetic_dashboard, "run_discovery", mine)
    app.run().button(key="gp_run").click().run()
    app.button(key="gp_run").click().run()
    assert not app.exception
    assert any("240" in message.value for message in app.error)
    assert not app.button(key="gp_run").disabled
    assert not any(button.key == "gp_save_df" for button in app.button)


def test_no_winners_is_valid_result_not_transform_error(app, monkeypatch):
    monkeypatch.setattr(genetic_dashboard, "run_discovery", Mock(return_value=result(False)))
    app.run().button(key="gp_run").click().run()
    assert not app.exception
    assert any("未发现" in warning.value for warning in app.warning)
    assert not any("降低" in warning.value for warning in app.warning)


def test_saving_error_is_visible(app, monkeypatch):
    monkeypatch.setattr(genetic_dashboard, "run_discovery", Mock(return_value=result()))
    monkeypatch.setattr(genetic_dashboard, "save_discovery", Mock(side_effect=OSError("disk full")))
    app.run().button(key="gp_run").click().run()
    app.button(key="gp_save_df").click().run()
    assert not app.exception
    assert any("disk full" in message.value for message in app.error)


def test_sparse_labeled_dates_are_disclosed(app, monkeypatch):
    discovered = result()
    discovered.report["data_quality"] = {"input_days": 751, "labeled_days": 620}
    monkeypatch.setattr(genetic_dashboard, "run_discovery", Mock(return_value=discovered))
    app.run().button(key="gp_run").click().run()
    assert not app.exception
    assert any("751" in warning.value and "620" in warning.value for warning in app.warning)


def test_failed_resave_does_not_report_success(app, monkeypatch, tmp_path):
    monkeypatch.setattr(genetic_dashboard, "run_discovery", Mock(return_value=result()))
    monkeypatch.setattr(genetic_dashboard, "save_discovery", Mock(side_effect=[tmp_path / "saved", OSError("disk full")]))
    app.run().button(key="gp_run").click().run()
    app.button(key="gp_save_df").click().run()
    app.button(key="gp_save_df").click().run()
    assert not app.exception
    assert any("disk full" in message.value for message in app.error)
    assert not any("已保存" in message.value for message in app.success)


def test_empty_input_is_reported_before_mining(app, monkeypatch):
    import os
    pd.DataFrame().to_pickle(os.environ["TEST_GP_FRAME"])
    mine = Mock()
    monkeypatch.setattr(genetic_dashboard, "run_discovery", mine)
    app.run().button(key="gp_run").click().run()
    assert not app.exception
    mine.assert_not_called()
    assert any("因子数据" in message.value for message in app.error)
