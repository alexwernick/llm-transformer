import pickle
import random
import re
import unicodedata


def normalise_line(line: str) -> tuple[str, str]:
    """Normalise a line of text and split into two at the tab character"""
    eng, fra = line.split("\t")

    def normalise_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text.strip().lower())
        text = re.sub(r"^([^ \w])(?!\s)", r" ", text)
        text = re.sub(r"(\s[^ \w])(?!\s)", r" ", text)
        text = re.sub(r"(?!\s)([^ \w])$", r" ", text)
        text = re.sub(r"(?!\s)([^ \w]\s)", r" ", text)
        return text

    eng = normalise_text(eng)
    fra = normalise_text(fra)
    fra = "[start] " + fra + " [end]"
    return eng, fra


def normalise_data(text_file: str) -> None:
    # normalize each line and separate into English and French
    with open(text_file) as fp:
        text_pairs: list[tuple[str, str]] = [normalise_line(line) for line in fp]

    # print some samples
    print("Printing some examples: ")
    for _ in range(5):
        print(random.choice(text_pairs))

    with open("text_pairs.pickle", "wb") as fp:
        pickle.dump(text_pairs, fp)
