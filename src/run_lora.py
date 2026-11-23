import argparse
import yaml
import torch
import time
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from data_utils import get_dataset, compute_metrics, preprocess_logits_for_metrics


# Кастомный коллбэк для профилирования: VRAM, градиенты и время
class ProfilingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()
        self.max_vram = 0.0
        self.max_grad_norm = 0.0

    def on_step_end(self, args, state, control, **kwargs):
        # Замер пиковой VRAM в гигабайтах
        if torch.cuda.is_available():
            current_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            if current_vram > self.max_vram:
                self.max_vram = current_vram

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Замер максимальной нормы градиента
        if logs and "grad_norm" in logs:
            if logs["grad_norm"] > self.max_grad_norm:
                self.max_grad_norm = logs["grad_norm"]

    def on_train_end(self, args, state, control, **kwargs):
        # Вывод итогов профилирования по окончании обучения
        total_time = time.time() - self.start_time
        print(f"\n--- Профилирование ---")
        print(f"Пиковая VRAM: {self.max_vram:.2f} GB")
        print(f"Максимальная норма градиента: {self.max_grad_norm:.4f}")
        print(f"Общее время обучения: {total_time:.2f} сек")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=int, required=True)
    args = parser.parse_args()

    # Читаем базовый конфиг
    with open("configs/_base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Настройка квантования (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    # Загрузка модели и токенизатора
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # Задаем pad_token, если его нет (частая проблема с базовыми моделями)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Настройка LoRA (r фиксирован, alpha меняется)
    lora_config = LoraConfig(
        r=cfg["r"],
        lora_alpha=args.alpha,
        target_modules=cfg["target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Загрузка данных (возвращает словарь с train и test)
    dataset_dict = get_dataset(cfg["dataset_name"])
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]

    # Настройка параметров обучения (SFTConfig)
    training_args = SFTConfig(
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
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    # Инициализация тренера с коллбэком профилирования
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[ProfilingCallback()]
    )

    print(f"Начинаем обучение с alpha={args.alpha}...")
    trainer.train()

    # Сохранение логов в CSV для Участника 2
    logs = trainer.state.log_history
    df = pd.DataFrame(logs)
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/logs_alpha_{args.alpha}.csv", index=False)
    print(f"Логи сохранены в results/logs_alpha_{args.alpha}.csv")


if __name__ == "__main__":
    main()