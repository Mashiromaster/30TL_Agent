# -*- coding: utf-8 -*-
# inference.py — Phase 2: 实时信号生成（训练/推理模式分离）

import pandas as pd
import os
import json
import tempfile

from model_release import ModelReleaseError, ReleaseManager
from strategy_runtime import ModelRuntime, SignalPolicy, get_strategy_spec


class SignalGenerator:
    """Backward-compatible facade over the canonical runtime and policy."""

    def __init__(
        self,
        model_path,
        strategy_id=None,
        *,
        runtime=None,
        policy=None,
        allow_legacy=False,
    ):
        model_path = os.path.abspath(os.fspath(model_path))
        base_dir = os.path.dirname(os.path.dirname(model_path))
        release_manager = ReleaseManager(
            os.path.join(base_dir, "artifacts", "models"),
            legacy_model_path=model_path,
        )
        signature = release_manager.current_signature()
        if runtime is None:
            if signature is None:
                raise ModelReleaseError("no model release is available")
            if signature.source != "release" and not allow_legacy:
                raise ModelReleaseError(
                    "unmanaged legacy pickle is disabled; train/promote a "
                    "verified release or opt in only for offline migration"
                )
            resolved_path = os.fspath(signature.artifact_path)
            self.runtime = ModelRuntime.from_path(resolved_path)
        else:
            self.runtime = runtime
        self.spec = get_strategy_spec(strategy_id)
        self.policy = policy or SignalPolicy(self.spec)
        self.model_release_id = (
            signature.release_id
            if signature is not None and signature.release_id
            else str(self.runtime.metadata.get("model_release_id", "legacy-unversioned"))
        )

        # Compatibility attributes for callers that inspect the old facade.
        self.model_base = self.runtime.model_base
        self.model_active = self.runtime.model_active
        self.scaler = self.runtime.scaler
        self.features = list(self.runtime.features)
        self.hierarchical_model = self.runtime.hierarchical_model
        self.selected_model_kind = self.runtime.selected_model_kind
        self.gp_miner = self.runtime.gp_miner

        print(
            f"[SignalGen] 模型已加载: {len(self.features)} 个特征; "
            f"strategy={self.spec.strategy_id}; release={self.model_release_id}"
        )

    def predict(self, df_factors):
        """Run model inference only; signal rules live in ``SignalPolicy``."""
        return self.runtime.predict_frame(df_factors)

    def generate_signal(self, df_factors):
        """
        Generate trading signal for the LATEST bar.
        Returns (signal_dict, df_with_predictions).
        """
        predictions = self.predict(df_factors)
        evaluated = self.policy.evaluate_frame(
            predictions, model_release_id=self.model_release_id
        )
        decision = self.policy.latest(
            predictions, model_release_id=self.model_release_id
        )
        signal = decision.to_dict()
        for key in (
            "predicted_return",
            "predicted_return_smooth",
            "confidence",
            "suggested_weight",
            "upper_threshold",
            "lower_threshold",
        ):
            if signal.get(key) is not None:
                signal[key] = round(float(signal[key]), 6 if "return" in key or "threshold" in key else 4)
        return signal, evaluated

    def print_signal(self, signal):
        """Pretty-print signal to console."""
        direction_mark = {1: '[LONG]', -1: '[SHORT]', 0: '[FLAT ]'}

        print("\n" + "=" * 50)
        print("         实时交易信号")
        print("=" * 50)
        print(f"  时间:       {signal['timestamp']}")
        print(f"  价格:       {signal['close']}")
        print(f"  市场状态:   {signal['regime_name']} (regime={signal['market_regime']})")
        print(f"  预测收益:   {signal['predicted_return']:.4f}%")
        print(f"  平滑预测:   {signal['predicted_return_smooth']:.4f}%")
        upper = signal.get('upper_threshold')
        lower = signal.get('lower_threshold')
        print(f"  上阈值:     {upper:.4f}%" if upper is not None else "  上阈值:     N/A")
        print(f"  下阈值:     {lower:.4f}%" if lower is not None else "  下阈值:     N/A")
        print(f"  " + "-" * 40)
        print(f"  信号方向:   {direction_mark.get(signal['direction'], '[????]')} {signal['direction_name']}")
        print(f"  置信度:     {signal['confidence']:.1%}")
        print(f"  建议仓位:   {signal['suggested_weight']:.1%}")
        print(f"  使用模型:   {signal['model_used']}")
        print("=" * 50)


def _atomic_json(destination, payload):
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_signal_history(destination, signal_out):
    row = pd.DataFrame([signal_out])
    if os.path.exists(destination):
        history = pd.read_csv(destination)
        history = pd.concat([history, row], ignore_index=True, sort=False)
    else:
        history = row
    if 'decision_id' in history:
        decision_ids = history['decision_id'].fillna('').astype(str)
        keep = decision_ids.eq('') | ~decision_ids.duplicated(keep='last')
        history = history.loc[keep]
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    try:
        history.to_csv(temporary, index=False, encoding='utf-8-sig')
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def run_inference(base_dir, *, strategy_id=None):
    """Full inference pipeline: factor extraction -> load model -> generate signal."""
    import factor_extraction

    MODEL_FILE = os.path.join(base_dir, "models/trained_model.pkl")
    FACTOR_FILE = os.path.join(base_dir, "outputs/df_factors.pkl")
    SIGNAL_JSON = os.path.join(base_dir, "outputs/signal.json")
    SIGNAL_CSV = os.path.join(base_dir, "outputs/signal_history.csv")

    # 1. Resolve either the active managed release or a read-only legacy model.
    manager = ReleaseManager(
        os.path.join(base_dir, "artifacts", "models"),
        legacy_model_path=MODEL_FILE,
    )
    signature = manager.current_signature()
    if signature is None or signature.source != "release":
        print("[Inference ERROR] 未找到通过门禁的托管模型 release")
        print("[Inference] 请先运行训练模式: python main.py --mode train")
        return False

    # 2. Run factor extraction (rebuilds df_factors.pkl)
    print("\n[1/3] 因子计算...")
    if not factor_extraction.run_process(base_dir, tick_subdir="data/tick"):
        print("[Inference ERROR] 因子计算失败")
        return False

    # 3. Load factors + model, generate signal
    print("\n[2/3] 加载模型并生成信号...")
    df_factors = pd.read_pickle(FACTOR_FILE)

    generator = SignalGenerator(MODEL_FILE, strategy_id=strategy_id)
    signal, df_pred = generator.generate_signal(df_factors)

    # 4. Output
    print("\n[3/3] 信号输出...")
    generator.print_signal(signal)

    # JSON output
    signal_out = {k: v for k, v in signal.items()}
    _atomic_json(SIGNAL_JSON, signal_out)
    print(f"\n[Inference] 信号JSON已保存: {SIGNAL_JSON}")

    # CSV history append
    _atomic_signal_history(SIGNAL_CSV, signal_out)
    print(f"[Inference] 信号历史已追加: {SIGNAL_CSV}")

    if not signal_out.get('decision_ready', True):
        print(
            "[Inference] 决策尚未就绪: "
            f"{signal_out.get('missing_reason', 'insufficient_history')}"
        )
        return False
    return True
