import torch
import torch.nn as nn
import torch.optim as optim

from llm_transformer.datacamp.transformer import Transformer


def main():
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512  # Higher values increase model capacity but require more computation
    num_heads = 8  # noqa: E501 More heads can capture diverse aspects of data, but are computationally intensive
    num_layers = (
        6  # More layers improve representation power, but can lead to overfitting
    )
    d_ff = 2048  # Larger feed-forward networks increase model robustness
    max_seq_length = 100
    dropout = 0.1  # Regularizes the model to prevent overfitting
    learning_rate = 0.0001  # Impacts convergence speed and stability
    # batch_size = 32  # noqa: E501 Larger batch sizes improve learning stability but require more memory
    # the fake data is actually simulating a batch size of 64
    epochs = 3  # 100

    transformer = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    )
    src_data, tgt_data = generate_data(src_vocab_size, tgt_vocab_size, max_seq_length)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    train(
        transformer,
        src_data,
        tgt_data,
        tgt_vocab_size,
        learning_rate,
        epochs,
        criterion,
    )
    evaluate_performance(
        transformer, criterion, src_vocab_size, tgt_vocab_size, max_seq_length
    )


def generate_data(src_vocab_size, tgt_vocab_size, max_seq_length):
    """
    For illustrative purposes, a dummy dataset will be crafted in this example.
    However, in a practical scenario, a more substantial dataset would be
    employed, and the process would involve text preprocessing along with the
    creation of vocabulary mappings for both the source and target languages.
    """
    src_data = torch.randint(
        1, src_vocab_size, (64, max_seq_length)
    )  # (batch_size, seq_length)
    tgt_data = torch.randint(
        1, tgt_vocab_size, (64, max_seq_length)
    )  # (batch_size, seq_length)
    return src_data, tgt_data


def train(
    transformer, src_data, tgt_data, tgt_vocab_size, learning_rate, epochs, criterion
):
    optimizer = optim.Adam(
        transformer.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
    )
    transformer.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


def evaluate_performance(
    transformer, criterion, src_vocab_size, tgt_vocab_size, max_seq_length
):
    transformer.eval()

    # Generate random sample validation data
    val_src_data = torch.randint(
        1, src_vocab_size, (64, max_seq_length)
    )  # (batch_size, seq_length)
    val_tgt_data = torch.randint(
        1, tgt_vocab_size, (64, max_seq_length)
    )  # (batch_size, seq_length)

    with torch.no_grad():
        val_output = transformer(val_src_data, val_tgt_data[:, :-1])
        val_loss = criterion(
            val_output.contiguous().view(-1, tgt_vocab_size),
            val_tgt_data[:, 1:].contiguous().view(-1),
        )
        print(f"Validation Loss: {val_loss.item()}")


if __name__ == "__main__":
    main()
