from typing import Union, List

from nltk import RegexpTokenizer
from transformers import BioGptTokenizer

from settings import MIMIC_MAX_LENGTH


class TextProcessor:
    regexp_tokenizer = RegexpTokenizer(r'\w+')
    biogpt_tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt')

    @classmethod
    def tokenize(cls, text: str):
        return cls.regexp_tokenizer.tokenize(text.lower())

    @classmethod
    def encode(cls, text: Union[List[str], str], padding: str = 'max_length', max_length: int = MIMIC_MAX_LENGTH,
               truncation: bool = True):
        return cls.biogpt_tokenizer.encode(text, padding=padding, max_length=max_length, truncation=truncation,
                                           return_tensors='pt')

    @classmethod
    def decode(cls, output):
        return cls.biogpt_tokenizer.decode(output, skip_special_tokens=True)


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
