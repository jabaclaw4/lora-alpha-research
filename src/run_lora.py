import argparse
import yaml
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from data_utils import get_dataset


def main():
    # Парсим аргумент alpha из командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=int, required=True)
    args = parser.parse_args()

    # Читаем базовый конфиг
    with open("configs/_base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Загружаем токенизатор и модель с квантованием в 4 бита для экономии памяти
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        load_in_4bit=cfg["load_in_4bit"],
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # Настраиваем LoRA. r фиксирован, alpha берем из аргументов
    lora_config = LoraConfig(
        r=cfg["r"],
        lora_alpha=args.alpha,
        target_modules=cfg["target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Подключаем LoRA к модели
    model = get_peft_model(model, lora_config)

    # Выводим количество обучаемых параметров для проверки
    model.print_trainable_parameters()

    # Готовим данные
    train_dataset = get_dataset(cfg["dataset_name"])

    # Настраиваем параметры тренировки
    training_args = TrainingArguments(
        output_dir=f"./results/checkpoints_alpha_{args.alpha}",
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        max_grad_norm=cfg["max_grad_norm"],
        save_strategy="no",
        report_to="none"  # Отключаем внешние логгеры для простоты
    )

    # Запускаем обучение
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
    )

    print(f"Начинаем обучение с alpha={args.alpha}...")
    trainer.train()

    # Важно: сохраняем логи в файл, чтобы Участник 2 потом собрал таблицу
    logs = trainer.state.log_history
    df = pd.DataFrame(logs)

    # Сохраняем в папку results
    import os
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/logs_alpha_{args.alpha}.csv", index=False)
    print(f"Логи сохранены в results/logs_alpha_{args.alpha}.csv")


if __name__ == "__main__":
    main()