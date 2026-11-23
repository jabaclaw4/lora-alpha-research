import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_TRAINABLE_PARAMS = 2179072  # Qwen2.5-1.5B-Instruct + LoRA r=8


def collect_metrics(results_dir: Path = RESULTS_DIR):
    """Собирает все logs_alpha_*.csv из папки results и сохраняет итоговый metrics.csv."""
    log_files = sorted(results_dir.glob("logs_alpha_*.csv"))

    if not log_files:
        logging.error(f"В папке {results_dir} нет файлов logs_alpha_*.csv. Сначала запустите обучение.")
        return

    rows = []

    for file_path in log_files:
        alpha_str = file_path.stem.replace("logs_alpha_", "")
        try:
            alpha = int(alpha_str)
        except ValueError:
            continue

        df = pd.read_csv(file_path)

        # Отделяем строки шагов обучения от строк валидации и финальной сводки
        is_train_row = pd.Series(True, index=df.index)
        if "eval_loss" in df.columns:
            is_train_row &= df["eval_loss"].isna()
        if "train_runtime" in df.columns:
            is_train_row &= df["train_runtime"].isna()

        train_df = df[is_train_row]

        # 1. Максимальная норма градиента
        grad_norm_max = None
        if "grad_norm" in train_df.columns and not train_df["grad_norm"].dropna().empty:
            grad_norm_max = float(train_df["grad_norm"].max())

        # 2. Проверка резкого скачка лосса (> 20% за шаг)
        loss_spike = False
        if "loss" in train_df.columns:
            losses = train_df["loss"].dropna()
            if len(losses) > 1:
                loss_spike = bool((losses.pct_change() > 0.20).any())

        # 3. Статус обучения: ok / clipped / diverged
        status = "ok"
        has_bad_loss = False
        if "loss" in train_df.columns and len(train_df) > 0:
            has_bad_loss = bool(train_df["loss"].isna().any() or np.isinf(train_df["loss"]).any())

        has_bad_grad = grad_norm_max is not None and (np.isnan(grad_norm_max) or np.isinf(grad_norm_max))

        if has_bad_loss or has_bad_grad:
            status = "diverged"
        elif grad_norm_max is not None and grad_norm_max > 10.0:
            status = "clipped"

        # 4. Финальные метрики валидации
        val_loss = None
        if "eval_loss" in df.columns and not df["eval_loss"].dropna().empty:
            val_loss = float(df["eval_loss"].dropna().iloc[-1])

        task_metric = None
        if "eval_accuracy" in df.columns and not df["eval_accuracy"].dropna().empty:
            task_metric = float(df["eval_accuracy"].dropna().iloc[-1])

        trainable_params = DEFAULT_TRAINABLE_PARAMS
        if "trainable_params" in df.columns and not df["trainable_params"].dropna().empty:
            trainable_params = int(df["trainable_params"].dropna().iloc[0])

        rows.append({
            "experiment": f"lora_r8_alpha_{alpha}",
            "alpha": alpha,
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "task_metric": round(task_metric, 4) if task_metric is not None else None,
            "grad_norm_max": round(grad_norm_max, 4) if grad_norm_max is not None else None,
            "loss_spike": loss_spike,
            "status": status,
            "trainable_params": trainable_params,
        })

    metrics_df = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    output_file = results_dir / "metrics.csv"
    metrics_df.to_csv(output_file, index=False)

    logging.info(f"Готово! Таблица сохранена в {output_file}:\n")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    collect_metrics()