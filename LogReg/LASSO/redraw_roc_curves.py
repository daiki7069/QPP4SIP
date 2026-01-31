"""
最新の結果（results.csv）からROC曲線を書き直すスクリプト。

各シナリオ（pre, post, pre_post, pre_post_bert, pre_post_roberta）について、
正則化4種（No penalty, L1, L2, ElasticNet）と個別指標を1枚のROCに描画。
- 統合モデルは「Model」ではなく正則化手法で表示（No penalty, L1, L2, ElasticNet）
- 凡例は大きく（画面の約4割）
- タイトルなし
- 縦・横軸のフォントを大きく
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
LASSO_DIR = Path(__file__).resolve().parent
if str(LASSO_DIR) not in sys.path:
    sys.path.insert(0, str(LASSO_DIR))

from module.visualization import plot_roc_curves_regularization_and_single_metrics

# AmbigNQ の最新結果のベースパス / ROC まとめ保存先
BASE = LASSO_DIR / "outputs" / "AmbigNQ"
OUTPUT_DIR = LASSO_DIR / "outputs" / "roc_curves"

SCENARIOS = [
    ("pre", BASE / "pre" / "ictf_idf_maxidf_scq_scs"),
    ("post", BASE / "post" / "clarity_ns50_nqc_smv_wig"),
    ("pre_post", BASE / "pre_post" / "ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig"),
    ("pre_post_bert", BASE / "pre_post_bert" / "ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig_bert"),
    ("pre_post_roberta", BASE / "pre_post_bert" / "ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig_rob"),
]

REG_COLUMNS = ["LogReg", "L1", "L2", "ENet"]
REG_LABELS = {"LogReg": "No penalty", "L1": "L1", "L2": "L2", "ENet": "ElasticNet"}


def load_and_plot(scenario_name: str, scenario_dir: Path, output_dir: Path) -> None:
    csv_path = scenario_dir / "csv" / "results.csv"

    if not csv_path.exists():
        print(f"  [SKIP] {scenario_name}: {csv_path} がありません")
        return

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        print(f"  [SKIP] {scenario_name}: 'label' 列がありません")
        return

    y_test = df["label"].values.astype(int)

    # 正則化4種の予測確率（存在する列のみ）
    regularization_probas = {}
    for col in REG_COLUMNS:
        if col in df.columns:
            regularization_probas[REG_LABELS[col]] = df[col].values.astype(float)

    if not regularization_probas:
        print(f"  [SKIP] {scenario_name}: 正則化列（LogReg, L1, L2, ENet）がありません")
        return

    # 個別指標 = label と正則化列以外
    exclude = {"label"} | set(REG_COLUMNS)
    single_cols = [c for c in df.columns if c not in exclude]
    single_metrics_df = df[single_cols].copy() if single_cols else pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 通常版（横長）
    output_path = output_dir / f"{scenario_name}.png"
    plot_roc_curves_regularization_and_single_metrics(
        y_test=y_test,
        regularization_probas=regularization_probas,
        single_metrics_df=single_metrics_df,
        output_path=output_path,
        axis_fontsize=22,
        legend_fontsize=20,
        legend_fraction=0.4,
        square=False,
    )
    print(f"  [OK] {scenario_name}: 保存先 {output_path}")

    # 別バージョン: 軸ラベル24pt・凡例をグラフ外・ラベルは (xxx) 形式・グラフは小さめ
    outside_path = output_dir / f"{scenario_name}_outside.png"
    plot_roc_curves_regularization_and_single_metrics(
        y_test=y_test,
        regularization_probas=regularization_probas,
        single_metrics_df=single_metrics_df,
        output_path=outside_path,
        axis_fontsize=24,
        legend_fontsize=20,
        legend_fraction=0.4,
        square=False,
        legend_outside=True,
        use_short_labels=True,
    )
    print(f"  [OK] {scenario_name} (凡例外): 保存先 {outside_path}")

    # _bert, _roberta は正方形版も追加出力
    if scenario_name in ("pre_post_bert", "pre_post_roberta"):
        square_path = output_dir / f"{scenario_name}_square.png"
        plot_roc_curves_regularization_and_single_metrics(
            y_test=y_test,
            regularization_probas=regularization_probas,
            single_metrics_df=single_metrics_df,
            output_path=square_path,
            axis_fontsize=22,
            legend_fontsize=20,
            legend_fraction=0.4,
            square=True,
        )
        print(f"  [OK] {scenario_name} (正方形): 保存先 {square_path}")


def main():
    print("ROC曲線を最新結果で書き直します（正則化4種 + 個別指標、凡例大・タイトルなし・軸フォント大）")
    print(f"保存先: {OUTPUT_DIR}")
    print("=" * 70)
    for name, dir_path in SCENARIOS:
        print(f"\n{name}: {dir_path}")
        load_and_plot(name, dir_path, OUTPUT_DIR)
    print("\n完了.")


if __name__ == "__main__":
    main()
