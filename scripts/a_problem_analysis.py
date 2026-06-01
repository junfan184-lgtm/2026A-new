from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA_FILE = PROJECT / "赛题A" / "附件1.原始数据.xlsx"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"


@dataclass
class ModelResult:
    task: str
    target: str
    model: str
    n_train: int
    n_test: int
    mae: float
    rmse: float
    r2: float


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE, header=1)
    df = df.rename(columns=lambda x: str(x).strip())
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for i in range(1, 19):
        south = f"{i}#风箱南侧温度"
        north = f"{i}#风箱北侧温度"
        out[f"{i}#风箱平均温度"] = out[[south, north]].mean(axis=1)

    pressure_cols = [f"{i}#风箱负压" for i in range(1, 19)]
    temp_avg_cols = [f"{i}#风箱平均温度" for i in range(1, 19)]
    out["风箱平均负压"] = out[pressure_cols].mean(axis=1)
    out["前段平均负压"] = out[[f"{i}#风箱负压" for i in range(1, 7)]].mean(axis=1)
    out["中段平均负压"] = out[[f"{i}#风箱负压" for i in range(7, 13)]].mean(axis=1)
    out["后段平均负压"] = out[[f"{i}#风箱负压" for i in range(13, 19)]].mean(axis=1)
    out["风箱平均温度"] = out[temp_avg_cols].mean(axis=1)
    out["前段平均温度"] = out[[f"{i}#风箱平均温度" for i in range(1, 7)]].mean(axis=1)
    out["中段平均温度"] = out[[f"{i}#风箱平均温度" for i in range(7, 13)]].mean(axis=1)
    out["后段平均温度"] = out[[f"{i}#风箱平均温度" for i in range(13, 19)]].mean(axis=1)
    out["大烟道平均负压"] = out[["1#大烟道负压", "2#大烟道负压"]].mean(axis=1)
    out["大烟道平均温度"] = out[["1#大烟道温度", "2#大烟道温度"]].mean(axis=1)
    return out


def save_basic_outputs(df: pd.DataFrame) -> dict:
    target = "烧结大烟道外排CO浓度"
    missing = df.isna().sum().sort_values(ascending=False)
    stats = df.describe().T
    stats.to_csv(OUTPUT_DIR / "01_基础统计.csv", encoding="utf-8-sig")
    missing.to_csv(OUTPUT_DIR / "02_缺失值统计.csv", encoding="utf-8-sig", header=["缺失数量"])
    df.to_csv(OUTPUT_DIR / "cleaned_data.csv", encoding="utf-8-sig", index=False)
    anomaly_mask = (df[target] < 1000) | (df[target] > 6000)
    df.loc[anomaly_mask, ["序列", target]].to_csv(
        OUTPUT_DIR / "02_CO异常候选点.csv",
        encoding="utf-8-sig",
        index=False,
    )

    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "co_mean": float(df[target].mean()),
        "co_std": float(df[target].std()),
        "co_min": float(df[target].min()),
        "co_max": float(df[target].max()),
        "co_q95": float(df[target].quantile(0.95)),
        "co_q99": float(df[target].quantile(0.99)),
        "co_lt1000_count": int((df[target] < 1000).sum()),
        "co_gt6000_count": int((df[target] > 6000).sum()),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def plot_overview(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", font="SimHei")
    plt.rcParams["axes.unicode_minus"] = False

    target = "烧结大烟道外排CO浓度"
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(df["序列"], df[target], color="#b91c1c", linewidth=1.2)
    axes[0].set_ylabel("CO浓度")
    axes[0].set_title("CO浓度时序变化")
    axes[1].plot(df["序列"], df["风箱平均负压"], color="#2563eb", linewidth=1.1)
    axes[1].set_ylabel("平均负压")
    axes[1].set_title("风箱平均负压时序变化")
    axes[2].plot(df["序列"], df["风箱平均温度"], color="#15803d", linewidth=1.1)
    axes[2].set_ylabel("平均温度")
    axes[2].set_xlabel("序列")
    axes[2].set_title("风箱平均温度时序变化")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_关键变量时序总览.png", dpi=180)
    plt.close(fig)

    key_cols = (
        ["烧结机机速L1设定"]
        + [f"{i}#风箱负压" for i in range(1, 19)]
        + [f"{i}#风箱平均温度" for i in range(1, 19)]
        + ["1#大烟道负压", "2#大烟道负压", "1#大烟道温度", "2#大烟道温度", target]
    )
    corr = df[key_cols].corr(numeric_only=True)
    corr.to_csv(OUTPUT_DIR / "03_关键变量相关系数矩阵.csv", encoding="utf-8-sig")

    co_corr = corr[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)
    co_corr.to_csv(OUTPUT_DIR / "04_CO相关性排序.csv", encoding="utf-8-sig", header=["与CO相关系数"])
    top = co_corr.head(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2563eb" if v < 0 else "#dc2626" for v in top.values]
    ax.barh(top.index, top.values, color=colors)
    ax.set_title("与CO浓度绝对相关性最高的20个变量")
    ax.set_xlabel("Pearson相关系数")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_CO相关性Top20.png", dpi=180)
    plt.close(fig)


def scan_lags(df: pd.DataFrame, max_lag: int = 80) -> pd.DataFrame:
    target = "烧结大烟道外排CO浓度"
    source_cols = (
        [f"{i}#风箱负压" for i in range(1, 19)]
        + [f"{i}#风箱平均温度" for i in range(1, 19)]
        + ["1#大烟道负压", "2#大烟道负压", "1#大烟道温度", "2#大烟道温度", "烧结机机速L1设定"]
    )
    rows = []
    for col in source_cols:
        for lag in range(max_lag + 1):
            corr = df[col].shift(lag).corr(df[target])
            rows.append({"变量": col, "滞后步数": lag, "滞后秒数": lag * 2, "相关系数": corr})
    lag_df = pd.DataFrame(rows)
    lag_df["绝对相关系数"] = lag_df["相关系数"].abs()
    lag_df.to_csv(OUTPUT_DIR / "05_变量滞后相关扫描.csv", encoding="utf-8-sig", index=False)

    best = lag_df.sort_values("绝对相关系数", ascending=False).groupby("变量", as_index=False).first()
    best = best.sort_values("绝对相关系数", ascending=False)
    best.to_csv(OUTPUT_DIR / "06_各变量最佳滞后相关.csv", encoding="utf-8-sig", index=False)

    top_vars = best.head(12)["变量"].tolist()
    fig, ax = plt.subplots(figsize=(12, 7))
    for col in top_vars:
        sub = lag_df[lag_df["变量"] == col]
        ax.plot(sub["滞后秒数"], sub["绝对相关系数"], label=col, linewidth=1.2)
    ax.set_title("Top变量的滞后相关强度曲线")
    ax.set_xlabel("滞后秒数")
    ax.set_ylabel("|相关系数|")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_滞后相关曲线Top变量.png", dpi=180)
    plt.close(fig)
    return best


def chronological_split(x: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8):
    n_train = int(len(x) * train_ratio)
    return x.iloc[:n_train], x.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]


def evaluate_model(task: str, target: str, name: str, model, x: pd.DataFrame, y: pd.Series) -> ModelResult:
    x_train, x_test, y_train, y_test = chronological_split(x, y)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return ModelResult(
        task=task,
        target=target,
        model=name,
        n_train=len(x_train),
        n_test=len(x_test),
        mae=float(mean_absolute_error(y_test, pred)),
        rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
        r2=float(r2_score(y_test, pred)),
    )


def make_temperature_features(df: pd.DataFrame, box_id: int, strategy: str) -> pd.DataFrame:
    pressure_cols = [f"{i}#风箱负压" for i in range(1, 19)]
    local_col = f"{box_id}#风箱负压"
    frames = [df["烧结机机速L1设定"].rename("烧结机机速L1设定")]

    if strategy == "local_lag":
        source_cols = [local_col]
    elif strategy == "neighbor_lag":
        neighbor_ids = [j for j in [box_id - 1, box_id, box_id + 1] if 1 <= j <= 18]
        source_cols = [f"{j}#风箱负压" for j in neighbor_ids]
        if box_id <= 6:
            frames.append(df["前段平均负压"].rename("前段平均负压"))
        elif box_id <= 12:
            frames.append(df["中段平均负压"].rename("中段平均负压"))
        else:
            frames.append(df["后段平均负压"].rename("后段平均负压"))
    elif strategy == "global_lag":
        source_cols = pressure_cols
        frames.extend(
            [
                df["前段平均负压"].rename("前段平均负压"),
                df["中段平均负压"].rename("中段平均负压"),
                df["后段平均负压"].rename("后段平均负压"),
            ]
        )
    elif strategy == "state_neighbor_lag":
        neighbor_ids = [j for j in [box_id - 1, box_id, box_id + 1] if 1 <= j <= 18]
        source_cols = [f"{j}#风箱负压" for j in neighbor_ids]
        temp_col = f"{box_id}#风箱平均温度"
        frames.extend(
            [
                df[temp_col].shift(1).rename(f"{temp_col}_lag1"),
                df[temp_col].shift(5).rename(f"{temp_col}_lag5"),
                df[temp_col].shift(10).rename(f"{temp_col}_lag10"),
            ]
        )
    else:
        raise ValueError(f"Unknown temperature feature strategy: {strategy}")

    for col in source_cols:
        frames.append(df[col].rename(col))
        for lag in [5, 10, 20]:
            frames.append(df[col].shift(lag).rename(f"{col}_lag{lag}"))
    return pd.concat(frames, axis=1)


def run_temperature_models(df: pd.DataFrame) -> list[ModelResult]:
    results: list[ModelResult] = []
    for i in range(1, 19):
        target = f"{i}#风箱平均温度"
        for strategy in ["local_lag", "neighbor_lag", "global_lag", "state_neighbor_lag"]:
            x = make_temperature_features(df, i, strategy)
            y = df[target]
            valid = x.notna().all(axis=1) & y.notna()
            x_valid, y_valid = x.loc[valid], y.loc[valid]
            models = {
                f"{strategy}_Ridge": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0)),
                f"{strategy}_GradientBoosting": GradientBoostingRegressor(
                    n_estimators=120,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=2026,
                    min_samples_leaf=5,
                ),
                f"{strategy}_RandomForest": RandomForestRegressor(
                    n_estimators=80,
                    max_depth=8,
                    random_state=2026,
                    n_jobs=1,
                    min_samples_leaf=5,
                ),
            }
            for name, model in models.items():
                results.append(evaluate_model("问题一_空间时滞温度预测", target, name, model, x_valid, y_valid))
    return results


def make_lagged_features(
    df: pd.DataFrame,
    lags: list[int],
    target_lags: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    target = "烧结大烟道外排CO浓度"
    base_cols = (
        ["烧结机机速L1设定"]
        + [f"{i}#风箱负压" for i in range(1, 19)]
        + [f"{i}#风箱平均温度" for i in range(1, 19)]
        + ["1#大烟道负压", "2#大烟道负压", "1#大烟道温度", "2#大烟道温度"]
    )
    feature_frames = []
    for col in base_cols:
        feature_frames.append(df[col].rename(col))
        for lag in lags:
            feature_frames.append(df[col].shift(lag).rename(f"{col}_lag{lag}"))
    for lag in target_lags or []:
        feature_frames.append(df[target].shift(lag).rename(f"{target}_lag{lag}"))
    x = pd.concat(feature_frames, axis=1)
    y = df[target]
    valid = x.notna().all(axis=1) & y.notna()
    return x.loc[valid], y.loc[valid]


def plot_prediction(df_plot: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_plot["序列"], df_plot["真实CO"], label="真实CO", color="#111827", linewidth=1.2)
    ax.plot(df_plot["序列"], df_plot["预测CO"], label="预测CO", color="#dc2626", linewidth=1.2)
    ax.set_title("CO浓度预测：测试集真实值 vs 预测值")
    ax.set_xlabel("序列")
    ax.set_ylabel("CO浓度")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_CO预测测试集对比.png", dpi=180)
    plt.close(fig)


def run_co_models(df: pd.DataFrame) -> tuple[list[ModelResult], pd.DataFrame, pd.DataFrame]:
    results: list[ModelResult] = []
    feature_sets = {
        "raw_lag0": (df, [], []),
        "raw_lag5_10_20": (df, [5, 10, 20], []),
        "filtered_lag0": (df[df["烧结大烟道外排CO浓度"].between(1000, 6000)].copy(), [], []),
        "filtered_lag5_10_20": (
            df[df["烧结大烟道外排CO浓度"].between(1000, 6000)].copy(),
            [5, 10, 20],
            [],
        ),
        "filtered_monitor_lag1_5_10": (
            df[df["烧结大烟道外排CO浓度"].between(1000, 6000)].copy(),
            [5, 10, 20],
            [1, 5, 10],
        ),
    }
    best_pred = None
    best_importance = None
    best_rmse = np.inf

    for feature_name, (model_df, lags, target_lags) in feature_sets.items():
        x, y = make_lagged_features(model_df, lags, target_lags)
        models = {
            f"{feature_name}_LinearRegression": make_pipeline(SimpleImputer(), StandardScaler(), LinearRegression()),
            f"{feature_name}_Ridge": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=50.0)),
            f"{feature_name}_GradientBoosting": GradientBoostingRegressor(
                n_estimators=180,
                learning_rate=0.04,
                max_depth=3,
                random_state=2026,
                min_samples_leaf=5,
            ),
            f"{feature_name}_RandomForest": RandomForestRegressor(
                n_estimators=120,
                max_depth=10,
                random_state=2026,
                n_jobs=1,
                min_samples_leaf=4,
            ),
        }
        for name, model in models.items():
            x_train, x_test, y_train, y_test = chronological_split(x, y)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            task_name = "问题二_CO监测预测" if target_lags else "问题二_CO调控预测"
            result = ModelResult(
                task=task_name,
                target="烧结大烟道外排CO浓度",
                model=name,
                n_train=len(x_train),
                n_test=len(x_test),
                mae=float(mean_absolute_error(y_test, pred)),
                rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
                r2=float(r2_score(y_test, pred)),
            )
            results.append(result)
            if result.rmse < best_rmse:
                best_rmse = result.rmse
                sequence = model_df.loc[x_test.index, "序列"].reset_index(drop=True)
                best_pred = pd.DataFrame({"序列": sequence, "真实CO": y_test.to_numpy(), "预测CO": pred})
                if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
                    importance = pd.DataFrame(
                        {"变量": x.columns, "重要性": model.feature_importances_}
                    ).sort_values("重要性", ascending=False)
                    best_importance = importance
                else:
                    best_importance = None

    if best_pred is None:
        raise RuntimeError("CO model did not produce predictions.")
    best_pred.to_csv(OUTPUT_DIR / "08_CO最佳模型测试集预测.csv", encoding="utf-8-sig", index=False)
    plot_prediction(best_pred)
    if best_importance is not None:
        best_importance.to_csv(OUTPUT_DIR / "09_CO随机森林特征重要性.csv", encoding="utf-8-sig", index=False)
        top = best_importance.head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.barh(top["变量"], top["重要性"], color="#0f766e")
        ax.set_title("CO预测模型特征重要性Top25")
        ax.set_xlabel("重要性")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "05_CO预测特征重要性Top25.png", dpi=180)
        plt.close(fig)
    return results, best_pred, best_importance


def run_pressure_optimization(df: pd.DataFrame) -> pd.DataFrame:
    target = "烧结大烟道外排CO浓度"
    filtered = df[df[target].between(1000, 6000)].copy()
    pressure_cols = [f"{i}#风箱负压" for i in range(1, 19)]
    feature_cols = (
        ["烧结机机速L1设定"]
        + pressure_cols
        + [f"{i}#风箱平均温度" for i in range(1, 19)]
        + ["1#大烟道负压", "2#大烟道负压", "1#大烟道温度", "2#大烟道温度"]
    )
    x = filtered[feature_cols]
    y = filtered[target]
    control_model = GradientBoostingRegressor(
        n_estimators=220,
        learning_rate=0.04,
        max_depth=3,
        random_state=2026,
        min_samples_leaf=5,
    )
    control_model.fit(x, y)

    bounds_df = pd.DataFrame(
        {
            "变量": pressure_cols,
            "历史最小值": filtered[pressure_cols].min().values,
            "5%分位": filtered[pressure_cols].quantile(0.05).values,
            "中位数": filtered[pressure_cols].median().values,
            "95%分位": filtered[pressure_cols].quantile(0.95).values,
            "历史最大值": filtered[pressure_cols].max().values,
        }
    )
    bounds_df.to_csv(OUTPUT_DIR / "10_风箱负压调控范围.csv", encoding="utf-8-sig", index=False)
    global_bounds = list(zip(bounds_df["5%分位"], bounds_df["95%分位"]))

    candidates = {
        "中位CO工况": int((filtered[target] - filtered[target].median()).abs().idxmin()),
        "高CO工况": int(filtered[target].idxmax()),
        "末端工况": int(filtered.index[-1]),
    }
    rows = []
    pressure_profiles = []
    for scenario, idx in candidates.items():
        base_row = filtered.loc[idx, feature_cols].copy()
        current_p = base_row[pressure_cols].to_numpy(dtype=float)
        scenario_bounds = []
        for p0, (lo, hi) in zip(current_p, global_bounds):
            center = min(max(p0, lo), hi)
            scenario_bounds.append((max(lo, center - 1.5), min(hi, center + 1.5)))
        ranges = np.array([hi - lo for lo, hi in scenario_bounds])

        def objective(p: np.ndarray) -> float:
            row = base_row.copy()
            row.loc[pressure_cols] = p
            pred = float(control_model.predict(pd.DataFrame([row], columns=feature_cols))[0])
            adjustment_penalty = 35.0 * float(np.mean(((p - current_p) / ranges) ** 2))
            smoothness_penalty = 8.0 * float(np.mean(np.diff(p) ** 2))
            return pred + adjustment_penalty + smoothness_penalty

        result = differential_evolution(
            objective,
            bounds=scenario_bounds,
            seed=2026,
            maxiter=80,
            popsize=8,
            polish=True,
            updating="immediate",
            workers=1,
        )
        optimized_p = result.x
        row_current = base_row.copy()
        row_optimized = base_row.copy()
        row_optimized.loc[pressure_cols] = optimized_p
        current_pred = float(control_model.predict(pd.DataFrame([row_current], columns=feature_cols))[0])
        optimized_pred = float(control_model.predict(pd.DataFrame([row_optimized], columns=feature_cols))[0])

        rows.append(
            {
                "工况": scenario,
                "序列": int(filtered.loc[idx, "序列"]),
                "真实CO": float(filtered.loc[idx, target]),
                "优化前预测CO": current_pred,
                "优化后预测CO": optimized_pred,
                "预测降低量": current_pred - optimized_pred,
                "预测降低比例": (current_pred - optimized_pred) / current_pred,
                "优化是否成功": bool(result.success),
                "优化目标值": float(result.fun),
            }
        )
        for col, before, after in zip(pressure_cols, current_p, optimized_p):
            pressure_profiles.append(
                {
                    "工况": scenario,
                    "风箱": col,
                    "当前负压": before,
                    "推荐负压": after,
                    "调整量": after - before,
                }
            )

    opt_df = pd.DataFrame(rows)
    profile_df = pd.DataFrame(pressure_profiles)
    opt_df.to_csv(OUTPUT_DIR / "11_典型工况CO优化效果.csv", encoding="utf-8-sig", index=False)
    profile_df.to_csv(OUTPUT_DIR / "12_典型工况推荐负压.csv", encoding="utf-8-sig", index=False)

    fig, ax = plt.subplots(figsize=(12, 7.5))
    for scenario in candidates:
        sub = profile_df[profile_df["工况"] == scenario]
        ax.plot(range(1, 19), sub["推荐负压"], marker="o", linewidth=3.0, markersize=9, label=scenario)
    ax.plot(range(1, 19), bounds_df["中位数"], color="#111827", linestyle="--", linewidth=2.6, label="历史中位数")
    ax.fill_between(range(1, 19), bounds_df["5%分位"], bounds_df["95%分位"], color="#93c5fd", alpha=0.28, label="5%-95%范围")
    ax.set_xlabel("风箱编号", fontsize=19)
    ax.set_ylabel("负压 KPa", fontsize=19)
    ax.tick_params(axis="both", labelsize=24)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 1, 3, 4]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=2, fontsize=22, frameon=True)
    ax.grid(True, linewidth=1.0, alpha=0.55)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_典型工况推荐负压曲线.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return opt_df


def write_report(summary: dict, lag_best: pd.DataFrame, results: pd.DataFrame, opt_df: pd.DataFrame) -> None:
    best_temp = (
        results[results["task"] == "问题一_空间时滞温度预测"]
        .sort_values(["target", "rmse"])
        .groupby("target", as_index=False)
        .first()
    )
    temp_strategy_summary = (
        results[results["task"] == "问题一_空间时滞温度预测"]
        .assign(特征方案=lambda d: d["model"].str.extract(r"^(state_neighbor_lag|neighbor_lag|local_lag|global_lag)")[0])
        .groupby("特征方案", as_index=False)
        .agg(平均MAE=("mae", "mean"), 平均RMSE=("rmse", "mean"), 平均R2=("r2", "mean"))
        .sort_values("平均RMSE")
    )
    monitor_co = results[results["task"] == "问题二_CO监测预测"].sort_values("rmse").head(5)
    control_co = results[results["task"] == "问题二_CO调控预测"].sort_values("rmse").head(5)
    lag_top = lag_best.head(10)

    report = [
        "# A题第二轮数据分析与建模报告",
        "",
        "## 1. 数据概况",
        f"- 样本数：{summary['rows']}，变量数（含工程特征）：{summary['cols']}。",
        f"- 缺失单元格：{summary['missing_cells']}。",
        f"- CO浓度均值：{summary['co_mean']:.2f}，标准差：{summary['co_std']:.2f}。",
        f"- CO浓度范围：{summary['co_min']:.2f} 至 {summary['co_max']:.2f}。",
        f"- CO浓度95%分位数：{summary['co_q95']:.2f}，99%分位数：{summary['co_q99']:.2f}。",
        f"- CO异常候选点：低于1000的样本 {summary['co_lt1000_count']} 个，高于6000的样本 {summary['co_gt6000_count']} 个。",
        "",
        "## 2. 滞后相关初步发现",
        lag_top[["变量", "滞后步数", "滞后秒数", "相关系数", "绝对相关系数"]].to_markdown(index=False),
        "",
        "## 3. 问题一：负压预测对应风箱温度",
        "先比较四类特征方案：局部负压滞后、相邻风箱空间滞后、全局风箱空间滞后、温度状态-相邻负压滞后。",
        temp_strategy_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "下表为每个风箱当前最优模型结果：",
        best_temp[["target", "model", "mae", "rmse", "r2"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 4. 问题二：CO浓度预测模型",
        "实时监测模型允许加入历史CO滞后项，适合在线预报与报警：",
        monitor_co[["model", "mae", "rmse", "r2"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "调控优化模型不加入历史CO项，避免优化目标被上一时刻CO浓度主导，适合后续负压决策：",
        control_co[["model", "mae", "rmse", "r2"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 5. 问题三：典型工况负压优化初步结果",
        opt_df[["工况", "序列", "真实CO", "优化前预测CO", "优化后预测CO", "预测降低量", "预测降低比例"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 6. 下一步建议",
        "- 问题一建议采用空间-时滞特征作为主模型，体现风箱沿烧结方向的耦合传播。",
        "- 问题二建议将实时监测模型与调控优化模型分开表述，避免历史CO滞后项造成决策解释混淆。",
        "- 问题三当前采用历史5%-95%分位与单次±1.5KPa调整约束，后续可继续增加相邻风箱平滑约束和分区协同约束。",
        "- 论文中可把异常CO段作为设备/检测异常或非稳态工况单独处理，并做稳健性讨论。",
        "",
    ]
    (OUTPUT_DIR / "A题第二轮分析报告.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = add_engineered_features(load_raw_data())
    summary = save_basic_outputs(df)
    plot_overview(df)
    lag_best = scan_lags(df)
    results = run_temperature_models(df)
    co_results, _, _ = run_co_models(df)
    results.extend(co_results)
    result_df = pd.DataFrame([r.__dict__ for r in results])
    result_df.to_csv(OUTPUT_DIR / "07_模型评价汇总.csv", encoding="utf-8-sig", index=False)
    opt_df = run_pressure_optimization(df)
    write_report(summary, lag_best, result_df, opt_df)


if __name__ == "__main__":
    main()
