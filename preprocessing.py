from nltk import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')


def preprocess_noteevent(note):
    tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
    return ' '.join(tokens)
