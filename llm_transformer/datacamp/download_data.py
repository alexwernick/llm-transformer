import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def load_opus_dataset(language_pair="en-es", debug_mode=False, debug_samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset with all its splits
    dataset_name = "Helsinki-NLP/opus_books" if debug_mode else "Helsinki-NLP/opus-100"

    dataset = load_dataset(dataset_name, language_pair)

    if debug_mode:
        for split in dataset:
            if len(dataset[split]) > debug_samples:
                dataset[split] = dataset[split].select(range(debug_samples))
        print(f"DEBUG MODE: Using {debug_samples} samples per split")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{language_pair}")

    # Extract source and target languages from the language pair
    src_lang, tgt_lang = language_pair.split("-")

    # Tokenize the data
    def tokenize_function(examples):
        # Source language tokenization
        source_texts = [ex[src_lang] for ex in examples["translation"]]
        source_encodings = tokenizer(
            source_texts, truncation=True, padding="max_length", return_tensors="pt"
        )

        # Target language tokenization
        target_texts = [ex[tgt_lang] for ex in examples["translation"]]
        target_encodings = tokenizer(
            target_texts, truncation=True, padding="max_length", return_tensors="pt"
        )

        return {
            "input_ids": source_encodings.input_ids,
            "labels": target_encodings.input_ids,
        }

    # Process each split separately
    tokenized_datasets = {}
    for split in dataset.keys():
        tokenized_datasets[split] = dataset[split].map(
            tokenize_function, batched=True, remove_columns=["translation"]
        )
        tokenized_datasets[split] = tokenized_datasets[split].with_format(
            "torch", device=device
        )

    # Return all processed splits
    return {
        "train": tokenized_datasets["train"] if "train" in tokenized_datasets else None,
        "validation": (
            tokenized_datasets["validation"]
            if "validation" in tokenized_datasets
            else None
        ),
        "test": tokenized_datasets["test"] if "test" in tokenized_datasets else None,
    }


# Example usage
"""
datasets = load_opus_dataset(debug_mode=True)
train_dataset = datasets["train"]
validation_dataset = datasets["validation"]
test_dataset = datasets["test"]

# Preview first few examples from training set
if train_dataset is not None:
    for i, example in enumerate(train_dataset):
        if i >= 3:  # Show just 3 examples
            break
        print(f"Example {i+1}:")
        print(f"Input IDs: {example['input_ids'].shape}")
        print(f"Labels: {example['labels'].shape}")
        print("-" * 40)
"""
