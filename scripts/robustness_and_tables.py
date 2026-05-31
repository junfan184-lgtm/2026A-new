from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "paper_tables"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def metric_row(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "方案": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def chronological_split(x: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8):
    n = int(len(x) * train_ratio)
    return x.iloc[:n], x.iloc[n:], y.iloc[:n], y.iloc[n:]


def build_co_features(df: pd.DataFrame, process_lags: list[int], co_lags: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    target = "烧结大烟道外排CO浓度"
    base_cols = (
        ["烧结机机速L1设定"]
        + [f"{i}#风箱负压" for i in range(1, 19)]
        + [f"{i}#风箱平均温度" for i in range(1, 19)]
        + ["1#大烟道负压", "2#大烟道负压", "1#大烟道温度", "2#大烟道温度"]
    )
    frames = []
    for col in base_cols:
        frames.append(df[col].rename(col))
        for lag in process_lags:
            frames.append(df[col].shift(lag).rename(f"{col}_lag{lag}"))
    for lag in co_lags:
        frames.append(df[target].shift(lag).rename(f"{target}_lag{lag}"))
    x = pd.concat(frames, axis=1)
    y = df[target]
    valid = x.notna().all(axis=1) & y.notna()
    return x.loc[valid], y.loc[valid]


def evaluate_model(name: str, model, x: pd.DataFrame, y: pd.Series) -> dict:
    x_train, x_test, y_train, y_test = chronological_split(x, y)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return metric_row(name, y_test, pred)


def make_robustness_outputs(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", font="SimHei")
    plt.rcParams["axes.unicode_minus"] = False
    target = "烧结大烟道外排CO浓度"
    raw = df.copy()
    filtered = df[df[target].between(1000, 6000)].copy()

    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=50)),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.04, max_depth=3, min_samples_leaf=5, random_state=2026
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=120, max_depth=10, min_samples_leaf=4, random_state=2026, n_jobs=1
        ),
    }

    anomaly_rows = []
    for data_name, data in [("原始数据", raw), ("剔除异常CO", filtered)]:
        x, y = build_co_features(data, process_lags=[], co_lags=[])
        for model_name, model in models.items():
            anomaly_rows.append(evaluate_model(f"{data_name}-{model_name}", model, x, y))
    anomaly_df = pd.DataFrame(anomaly_rows)
    anomaly_df.to_csv(TABLE_DIR / "表_异常处理稳健性.csv", encoding="utf-8-sig", index=False)

    lag_rows = []
    lag_settings = {
        "无滞后": [],
        "10s/20s/40s": [5, 10, 20],
        "20s/40s/80s": [10, 20, 40],
        "40s/80s/120s": [20, 40, 60],
    }
    for lag_name, lags in lag_settings.items():
        x, y = build_co_features(filtered, process_lags=lags, co_lags=[])
        model = GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.04, max_depth=3, min_samples_leaf=5, random_state=2026
        )
        lag_rows.append(evaluate_model(lag_name, model, x, y))
    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(TABLE_DIR / "表_工艺变量滞后敏感性.csv", encoding="utf-8-sig", index=False)

    monitor_rows = []
    monitor_settings = {
        "仅工艺变量": ([], []),
        "工艺滞后": ([5, 10, 20], []),
        "CO_lag1": ([5, 10, 20], [1]),
        "CO_lag1_5": ([5, 10, 20], [1, 5]),
        "CO_lag1_5_10": ([5, 10, 20], [1, 5, 10]),
    }
    for setting, (process_lags, co_lags) in monitor_settings.items():
        x, y = build_co_features(filtered, process_lags=process_lags, co_lags=co_lags)
        model = RandomForestRegressor(
            n_estimators=120, max_depth=10, min_samples_leaf=4, random_state=2026, n_jobs=1
        )
        monitor_rows.append(evaluate_model(setting, model, x, y))
    monitor_df = pd.DataFrame(monitor_rows)
    monitor_df.to_csv(TABLE_DIR / "表_CO历史滞后项敏感性.csv", encoding="utf-8-sig", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, data, title in [
        (axes[0], anomaly_df, "异常处理稳健性"),
        (axes[1], lag_df, "工艺滞后敏感性"),
        (axes[2], monitor_df, "CO历史项敏感性"),
    ]:
        sns.barplot(data=data, x="方案", y="RMSE", ax=ax, color="#2563eb")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_稳健性分析RMSE对比.png", dpi=180)
    plt.close(fig)


def make_paper_tables(df: pd.DataFrame) -> None:
    target = "烧结大烟道外排CO浓度"
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    model_results = pd.read_csv(OUTPUT_DIR / "07_模型评价汇总.csv")
    lag_best = pd.read_csv(OUTPUT_DIR / "06_各变量最佳滞后相关.csv")
    opt_effect = pd.read_csv(OUTPUT_DIR / "11_典型工况CO优化效果.csv")
    opt_pressure = pd.read_csv(OUTPUT_DIR / "12_典型工况推荐负压.csv")
    bounds = pd.read_csv(OUTPUT_DIR / "10_风箱负压调控范围.csv")

    variable_table = pd.DataFrame(
        [
            ["序列", "时间索引", "相邻样本间隔2s", "时间序列定位"],
            ["烧结机机速L1设定", "m/min", "控制变量", "料层停留时间调节"],
            ["1#-18#风箱负压", "KPa", "控制变量", "抽风强度与供氧状态"],
            ["1#-18#风箱平均温度", "摄氏度", "状态变量", "南北温度均值"],
            ["1#/2#大烟道负压", "KPa", "状态变量", "烟气汇集抽风状态"],
            ["1#/2#大烟道温度", "摄氏度", "状态变量", "烟道热工状态"],
            ["烧结大烟道外排CO浓度", "mg/m3", "目标变量", "排放预测与优化目标"],
        ],
        columns=["变量", "单位", "类型", "建模含义"],
    )
    variable_table.to_csv(TABLE_DIR / "表1_变量说明.csv", encoding="utf-8-sig", index=False)

    stats_table = pd.DataFrame(
        [
            ["样本数量", summary["rows"]],
            ["建模变量数（含工程特征）", summary["cols"]],
            ["缺失单元格", summary["missing_cells"]],
            ["CO均值", round(summary["co_mean"], 3)],
            ["CO标准差", round(summary["co_std"], 3)],
            ["CO最小值", round(summary["co_min"], 3)],
            ["CO最大值", round(summary["co_max"], 3)],
            ["CO低于1000样本数", summary["co_lt1000_count"]],
            ["CO高于6000样本数", summary["co_gt6000_count"]],
        ],
        columns=["统计项", "数值"],
    )
    stats_table.to_csv(TABLE_DIR / "表2_数据概况与异常统计.csv", encoding="utf-8-sig", index=False)

    temp_results = model_results[model_results["task"].str.contains("温度", na=False)].copy()
    temp_results["特征方案"] = temp_results["model"].str.extract(
        r"^(state_neighbor_lag|neighbor_lag|local_lag|global_lag)"
    )[0]
    temp_summary = (
        temp_results.groupby("特征方案", as_index=False)
        .agg(平均MAE=("mae", "mean"), 平均RMSE=("rmse", "mean"), 平均R2=("r2", "mean"))
        .sort_values("平均RMSE")
    )
    temp_summary.to_csv(TABLE_DIR / "表3_问题一温度预测模型对比.csv", encoding="utf-8-sig", index=False)

    best_temp = temp_results.sort_values(["target", "rmse"]).groupby("target", as_index=False).first()
    best_temp[["target", "model", "mae", "rmse", "r2"]].to_csv(
        TABLE_DIR / "表4_各风箱最优温度预测结果.csv", encoding="utf-8-sig", index=False
    )

    co_monitor = model_results[model_results["task"].str.contains("监测", na=False)].sort_values("rmse")
    co_control = model_results[model_results["task"].str.contains("调控", na=False)].sort_values("rmse")
    co_model_table = pd.concat(
        [
            co_monitor.assign(模型用途="实时监测").head(5),
            co_control.assign(模型用途="调控优化").head(5),
        ],
        axis=0,
    )[["模型用途", "model", "mae", "rmse", "r2"]]
    co_model_table.to_csv(TABLE_DIR / "表5_CO预测模型对比.csv", encoding="utf-8-sig", index=False)

    lag_best.head(15).to_csv(TABLE_DIR / "表6_滞后相关Top15.csv", encoding="utf-8-sig", index=False)
    opt_effect.to_csv(TABLE_DIR / "表7_典型工况优化效果.csv", encoding="utf-8-sig", index=False)
    bounds.to_csv(TABLE_DIR / "表8_风箱负压调控范围.csv", encoding="utf-8-sig", index=False)

    pivot = opt_pressure.pivot(index="风箱", columns="工况", values="推荐负压").reset_index()
    pivot.to_csv(TABLE_DIR / "表9_典型工况推荐负压汇总.csv", encoding="utf-8-sig", index=False)

    markdown_parts = [
        "# A题论文表格索引",
        "",
        "## 表1 变量说明",
        variable_table.to_markdown(index=False),
        "",
        "## 表2 数据概况与异常统计",
        stats_table.to_markdown(index=False),
        "",
        "## 表3 问题一温度预测模型对比",
        temp_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 表5 CO预测模型对比",
        co_model_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 表7 典型工况优化效果",
        opt_effect.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (TABLE_DIR / "论文表格索引.md").write_text("\n".join(markdown_parts), encoding="utf-8")

    with pd.ExcelWriter(TABLE_DIR / "A题论文表格汇总.xlsx", engine="openpyxl") as writer:
        for csv_file in sorted(TABLE_DIR.glob("表*.csv")):
            sheet = csv_file.stem[:31]
            pd.read_csv(csv_file).to_excel(writer, sheet_name=sheet, index=False)


def write_robustness_report() -> None:
    anomaly = pd.read_csv(TABLE_DIR / "表_异常处理稳健性.csv")
    lag = pd.read_csv(TABLE_DIR / "表_工艺变量滞后敏感性.csv")
    monitor = pd.read_csv(TABLE_DIR / "表_CO历史滞后项敏感性.csv")

    lines = [
        "# A题稳健性分析补充报告",
        "",
        "## 1. 异常处理稳健性",
        anomaly.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 2. 工艺变量滞后阶数敏感性",
        lag.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 3. CO历史滞后项敏感性",
        monitor.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 4. 论文写作建议",
        "- 异常CO段应作为非稳态或检测异常候选点单独说明，主模型采用剔除异常后的稳健结果。",
        "- 工艺变量滞后项对调控模型提升有限，说明CO排放受多因素和未观测工况影响，论文中应强调模型用于辅助决策而非单点精确控制。",
        "- 加入CO历史滞后项后监测模型显著提升，可作为在线预警模型；调控优化仍应采用不含历史CO项的模型，以保证控制变量解释性。",
        "",
    ]
    (OUTPUT_DIR / "A题稳健性分析补充报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(OUTPUT_DIR / "cleaned_data.csv", encoding="utf-8-sig")
    make_robustness_outputs(df)
    make_paper_tables(df)
    write_robustness_report()


if __name__ == "__main__":
    main()
