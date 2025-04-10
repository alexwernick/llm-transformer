import pathlib

import tensorflow as tf


def download_data() -> str:
    # download dataset provided by Anki: https://www.manythings.org/anki/
    # Define the file name and URL
    filename = "fra-eng.zip"
    url = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"

    # Download and extract the dataset
    text_file = tf.keras.utils.get_file(
        fname=filename,
        origin=url,
        extract=True,
    )  # will not download if file already exists
    # Update the path to the extracted text file
    text_file = pathlib.Path(text_file) / "fra.txt"
    print(f"File downloaded and extracted to: {text_file}")
    return text_file
