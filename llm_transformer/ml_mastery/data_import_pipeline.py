from llm_transformer.ml_mastery.data_download import download_data
from llm_transformer.ml_mastery.normalise_data import normalise_data

file_name: str = download_data()
normalise_data(file_name)
