```markdown
# llm-transformer

This repository contains an implementation of a Transformer-based model for melody generation. The project demonstrates how to preprocess musical data, train a Transformer model, and generate melodies using the trained model. The implementation is modular, with each component organized into separate files for clarity and reusability.

The project is inspired by the YouTube video [19. Melody generation with transformers - Generative Music AI"](https://www.youtube.com/watch?v=j4LABY2d7k4&t=309s) and is designed to provide a hands-on understanding of using Transformers for sequence-to-sequence tasks in the domain of generative music.

---

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
  - [1. `transformer.py`](#1-transformerpy)
  - [2. `train.py`](#2-trainpy)
  - [3. `melody_preprocessor.py`](#3-melody_preprocessorpy)
  - [4. `melody_generator.py`](#4-melody_generatorpy)
  - [5. `dataset.json`](#5-datasetjson)
- [Usage](#usage)
- [Installation](#installation)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project implements a Transformer model to generate melodies. The pipeline includes:

1. **Preprocessing musical data**: Tokenizing and encoding melodies into sequences suitable for model training.
2. **Training a Transformer model**: Training the model on a dataset of melodies to learn sequence-to-sequence relationships.
3. **Generating melodies**: Using the trained Transformer to generate new musical sequences based on a starting seed.

---

## File Structure

### 1. `transformer.py`

This file contains the implementation of the Transformer model. The implementation leverages **TensorFlow** to build the model components, trainable layers, and computations. Key highlights of TensorFlow usage include:

- **Custom Layers**: The Encoder, Decoder, and their respective subcomponents (e.g., `EncoderLayer`, `DecoderLayer`) are implemented as custom `tf.keras.layers.Layer` classes.
- **Multi-Head Attention**: TensorFlow's `MultiHeadAttention` layer is used to compute self-attention and cross-attention.
- **Feed-Forward Networks**: Built using TensorFlow's `Dense` layers.
- **Normalization and Dropout**: TensorFlow's `LayerNormalization` and `Dropout` are used to regularize and stabilize training.
- **Positional Encoding**: A custom implementation of sinusoidal positional encoding provides the model with information about the relative position of tokens.
- **Integration with TensorFlow Models**: The `Transformer` class inherits from `tf.keras.Model`, enabling seamless use of TensorFlow's training workflows and utilities.

The file includes the following components:

- **`Transformer`**: Combines the Encoder and Decoder to implement the full Transformer architecture.
- **`Encoder` and `Decoder`**: Handle input and target sequences, respectively.
- **`EncoderLayer` and `DecoderLayer`**: The core building blocks of the Transformer, featuring multi-head attention and feed-forward layers.
- **`sinusoidal_position_encoding`**: Provides positional information to the model, enabling it to understand the order of tokens in a sequence.

This modular implementation allows flexibility for adapting the Transformer to various sequence-to-sequence tasks beyond melody generation.

---

### 2. `train.py`

This file defines the training pipeline, including:

- **Loss Function**: Implements a Sparse Categorical Crossentropy loss function, with masking to ignore padded values.
- **Training Step**: Executes a single training step, including forward pass, backpropagation, and parameter updates.
- **Training Loop**: Iterates over multiple epochs, updating the model using the provided dataset.

The training process uses a dataset of melodies and the Transformer model defined in `transformer.py`.

---

### 3. `melody_preprocessor.py`

This file contains the `MelodyPreprocessor` class, responsible for preparing the melody dataset. Key features include:

- **Tokenization and Encoding**: Converts melodies into tokenized sequences.
- **Input-Target Pairs**: Creates input and target sequences for sequence-to-sequence training.
- **Padding**: Ensures all sequences are padded to a uniform length for batch processing.
- **Dataset Creation**: Converts the processed data into a TensorFlow `Dataset` for use in training.

The dataset must be in JSON format, with each melody represented as a string of comma-separated notes.

---

### 4. `melody_generator.py`

This file defines the `MelodyGenerator` class, which uses a trained Transformer model to generate melodies. Key functionality includes:

- **Melody Generation**: Iteratively predicts and appends notes to a given seed sequence until the desired melody length is reached.
- **Tokenization and Decoding**: Encodes the input seed sequence and decodes the generated sequence back into human-readable note notation.

The `generate` method takes a starting seed sequence and produces a complete melody.

---

### 5. `dataset.json`

This file contains the dataset of melodies used for training. Each melody is represented as a string of notes in the format:

```
"Note-Octave-Duration"
```

For example, a melody might look like:

```
"C4-1.0, D4-1.0, E4-1.0, F4-1.0"
```

The dataset is loaded and processed by the `MelodyPreprocessor` class.

---

## Usage

### 1. Preprocess the Dataset

Run the `melody_preprocessor.py` script to tokenize and encode the melodies in `dataset.json`. This will create a TensorFlow dataset for training.

```bash
python melody_preprocessor.py
```

### 2. Train the Model

Use the `train.py` script to train the Transformer model on the preprocessed dataset.

```bash
python train.py
```

### 3. Generate a Melody

After training, use the `melody_generator.py` script to generate a melody based on a starting seed sequence.

```bash
python melody_generator.py
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/melody-generation-transformer.git
   cd melody-generation-transformer
   ```

2. Install the required dependencies:

   ```bash
   poetry install
   ```

3. Ensure `dataset.json` is in the root directory.

---

## Acknowledgements

This project is inspired by the YouTube video[19. Melody generation with transformers - Generative Music AI"](https://www.youtube.com/watch?v=j4LABY2d7k4&t=309s). Special thanks to [Valerio Velardo](https://www.youtube.com/@ValerioVelardoTheSoundofAI) for providing a detailed explanation of Transformer-based melody generation.

