import logging
import numpy as np
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_dataset(
    dataset_name: str,
    max_rows: int = 1000,
    val_ratio: float = 0.1,
    seed: int = 42,
    eos_token: str = "",
):
    """Скачивает датасет, форматирует промпты и делит на train/val."""
    logging.info(f"Загружаем датасет: {dataset_name}")
    raw_dataset = load_dataset(dataset_name, split="train")

    if len(raw_dataset) > max_rows:
        raw_dataset = raw_dataset.select(range(max_rows))

    columns = set(raw_dataset.column_names)

    def format_row(row):
        # Формат Alpaca (instruction / input / output)
        if "instruction" in columns and "output" in columns:
            instruction = str(row.get("instruction", "")).strip()
            extra_input = str(row.get("input", "")).strip()
            answer = str(row.get("output", "")).strip()

            if extra_input:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{extra_input}\n\n### Response:\n{answer}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{answer}"
        # Уже готовая колонка text
        elif "text" in columns:
            text = str(row.get("text", "")).strip()
        # Запасной вариант для любой другой структуры колонок
        else:
            text = " ".join(str(v).strip() for v in row.values() if v)

        if eos_token and text and not text.endswith(eos_token):
            text += eos_token

        return {"text": text}

    dataset = raw_dataset.map(format_row, remove_columns=list(columns))
    dataset = dataset.filter(lambda row: len(row["text"].strip()) > 10)

    return dataset.train_test_split(test_size=val_ratio, seed=seed)


def preprocess_logits_for_metrics(logits, labels):
    """Считает argmax сразу на GPU, чтобы валидация не уронила Colab по RAM."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    """Считает точность предсказания следующего токена (Next-Token Accuracy)."""
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]
    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    # Сдвиг на 1 токен для Causal LM
    shift_preds = preds[:, :-1]
    shift_labels = labels[:, 1:]

    # Игнорируем паддинг (-100)
    mask = shift_labels != -100
    if not np.any(mask):
        return {"accuracy": 0.0}

    return {"accuracy": float(np.mean(shift_preds[mask] == shift_labels[mask]))}