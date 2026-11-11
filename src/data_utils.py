from datasets import load_dataset


def get_dataset(dataset_name: str):
    #$загружае датасет из хаба HuggingFace
    dataset = load_dataset(dataset_name, split="train")

    #берем только небольшую часть для тестов, чтобы обучение шло быстро
    dataset = dataset.select(range(min(1000, len(dataset))))

    #тут Участник 2 должен будет добавить форматирование(prompt template)

    def format_prompt(example):
        return {"text": example["instruction"] + " " + example.get("output", "")}

    dataset = dataset.map(format_prompt)
    return dataset