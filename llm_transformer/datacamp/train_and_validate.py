import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llm_transformer.datacamp.download_data import load_opus_dataset
from llm_transformer.datacamp.transformer import Transformer


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    d_model = 120  # Higher values increase model capacity but require more computation
    num_heads = 8  # noqa: E501 More heads can capture diverse aspects of data, but are computationally intensive
    num_layers = (
        6  # More layers improve representation power, but can lead to overfitting
    )
    d_ff = 1000  # Larger feed-forward networks increase model robustness
    max_seq_length = 512  # I have fixed this to match what we get back from
    # tokenizer but there must be a better way
    dropout = 0.1  # Regularizes the model to prevent overfitting
    learning_rate = 0.0001  # Impacts convergence speed and stability
    batch_size = 10  # noqa: E501 Larger batch sizes improve learning stability but require more memory
    epochs = 3  # 100

    # src_data, tgt_data = generate_data(src_vocab_size, tgt_vocab_size, max_seq_length)
    language_pair = "en-es"
    datasets = load_opus_dataset(language_pair, debug_mode=False, debug_samples=10)

    tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{language_pair}")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(device)

    transformer = Transformer(
        tokenizer.vocab_size,
        tokenizer.vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
        device=device,
    )

    train(
        transformer,
        datasets,
        tokenizer.vocab_size,
        learning_rate,
        epochs,
        criterion,
        batch_size,
        device,
    )
    evaluate_performance(
        transformer, datasets, criterion, tokenizer.vocab_size, batch_size, device
    )


def train(
    transformer,
    datasets,
    tgt_vocab_size,
    learning_rate,
    epochs,
    criterion,
    batch_size,
    device,
):
    train_dataloader = DataLoader(
        datasets["train"], batch_size=batch_size, shuffle=True
    )

    optimizer = optim.Adam(
        transformer.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
    )

    transformer.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            # Extract source and target data and move to device
            src_data = batch["input_ids"].to(device)
            tgt_data = batch["labels"].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass, but only input the target tokens up to the last one
            output = transformer(
                src_data,
                tgt_data[:, :-1],
            )

            # Calculate loss (reshape output and targets for cross-entropy)
            loss = criterion(
                output.contiguous().view(-1, tgt_vocab_size),
                tgt_data[:, 1:].contiguous().view(-1),
            )

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Track loss
            total_loss += loss.item()

            # Print batch progress
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        # Print epoch progress
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")


def evaluate_performance(
    transformer, datasets, criterion, tgt_vocab_size, batch_size, device
):
    val_dataloader = (
        DataLoader(datasets["validation"], batch_size=batch_size)
        if datasets["validation"] is not None
        else None
    )

    if not val_dataloader:
        print("There is no validation dataset")
        return

    transformer.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_dataloader:
            # Extract source and target data and move to device
            src_data = batch["input_ids"].to(device)
            tgt_data = batch["labels"].to(device)

            # Forward pass
            output = transformer(
                src_data,
                tgt_data[:, :-1],
            )

            # Calculate loss
            loss = criterion(
                output.contiguous().view(-1, tgt_vocab_size),
                tgt_data[:, 1:].contiguous().view(-1),
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
