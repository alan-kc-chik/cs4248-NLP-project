import pandas as pd
from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from spellchecker import SpellChecker
from nltk import pos_tag
import string

nltk_stopwords = list(stopwords.words('english'))


class Preprocess:
    def __init__(self, tokenizer=None, stemmer=None, lemmatizer=None,
                 lowercase=True, remove_stopwords=True, remove_punctuation=True, remove_number=True, spell_check=True,
                 input_filename=None, output_filename=None):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.lemmatizer = lemmatizer
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.remove_number = remove_number
        self.spell_check = spell_check
        self.input_filename = input_filename
        self.output_filename = output_filename

    def process(self):
        data, labels = self.read_data(self.input_filename)
        self._process(data, labels)

    def read_data(self, filename):
        raw_data = pd.read_csv(filename, header=None)
        data, labels = raw_data[1].tolist(), raw_data[0].tolist()
        return data, labels

    def _process(self, data, labels):
        if self.tokenizer == None:
            raise ValueError('Error.')
        dic = {'text': [], 'label': []}
        spell = SpellChecker()
        for text, label in zip(data, labels):
            token_list = self.tokenizer.tokenize(text)
            if self.spell_check:
                token_list = [spell.correction(token) for token in token_list]
                token_list = [token for token in token_list if token is not None and len(token) > 0]
            if self.lemmatizer is not None:
                token_list = self.lemmatize(self.lemmatizer, token_list)
            elif self.stemmer is not None:
                token_list = self.stem(self.stemmer, token_list)

            if self.lowercase:
                token_list = [token.lower() for token in token_list]
            if self.remove_stopwords:
                token_list = [token for token in token_list if token not in nltk_stopwords]
            if self.remove_punctuation:
                token_list = [''.join(ch for ch in token if ch not in string.punctuation) for token in token_list]
                token_list = [token for token in token_list if len(token) > 0]
            if self.remove_number:
                token_list = [''.join(ch for ch in token if ch not in [chr(48 + i) for i in range(10)]) for token in
                              token_list]
                token_list = [token for token in token_list if len(token) > 0]

            new_text = ' '.join(token_list)
            dic['text'].append(new_text)
            dic['label'].append(label)
            if len(dic['text']) % 1000 == 0:
                print(len(dic['text']), f'/ {len(data)}')
        self.generate(dic)

    def lemmatize(self, lemmatizer, token_list):
        pos_tag_list = pos_tag(token_list)
        for idx, (token, tag) in enumerate(pos_tag_list):
            tag_simple = tag[0].lower()
            if tag_simple in ['n', 'v', 'j']:
                word_type = tag_simple.replace('j', 'a')
            else:
                word_type = 'n'
            lemmatized_token = lemmatizer.lemmatize(token, pos=word_type)
            token_list[idx] = lemmatized_token
        return token_list

    def stem(self, stemmer, token_list):
        for idx, token in enumerate(token_list):
            token_list[idx] = stemmer.stem(token)
        return token_list

    def generate(self, dic):
        df = pd.DataFrame(dic)
        df.to_csv(self.output_filename)


tweet_tokenizer = TweetTokenizer()
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

process = Preprocess(tokenizer=tweet_tokenizer, lemmatizer=wordnet_lemmatizer, spell_check=False, output_filename='./process_train.csv',
                    input_filename='./dataset/raw_data/fulltrain.csv')
process.process()
process = Preprocess(tokenizer=tweet_tokenizer, lemmatizer=wordnet_lemmatizer, spell_check=False, output_filename='./process_test.csv',
                    input_filename='./dataset/raw_data/fulltrain.csv')
process.process()