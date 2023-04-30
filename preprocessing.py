from typing import Union, List

from nltk import RegexpTokenizer
from transformers import BioGptTokenizer

from settings import MIMIC_MAX_LENGTH


class Preprocessor:
    regexp_tokenizer = RegexpTokenizer(r'\w+')
    biogpt_tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt')

    @classmethod
    def tokenize(cls, text: str):
        return cls.regexp_tokenizer.tokenize(text.lower())

    @classmethod
    def encode(cls, text: Union[List[str], str], padding='max_length'):
        return cls.biogpt_tokenizer.encode(text, padding=padding, return_tensors='pt')


def sliding_window(input_array, window_size=512, stride=256):
    output_arrays = []
    for i in range(0, len(input_array), stride):
        window = input_array[i:i + window_size]
        if len(window) == window_size:
            output_arrays.append(window)
        elif len(window) > 0:
            output_arrays.append(window)
    return output_arrays


if __name__ == '__main__':
    pass
