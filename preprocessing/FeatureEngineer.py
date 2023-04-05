import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import csv


class FeatureEngineer:
    def __init__(self, tfidf=True, countvec=False, number_of_words=False, number_of_exclamatory=False, number_of_numbers=False):
        self.tfidf = tfidf
        self.countvec = countvec
        self.number_of_words = number_of_words
        self.number_of_exclamatory = number_of_exclamatory
        self.number_of_numbers = number_of_numbers
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=0.15)
        self.count_vectorizer = CountVectorizer(max_df=0.85, min_df=0.15)

    def encode(self, process_input_filename=None, original_input_filename=None, output_filename=None, istrain=True):
        data, labels = self.read_data(process_input_filename, col1='text', col2='label')
        odata, olabels = self.read_data(original_input_filename, col1=1, col2=0, header=False)
        combination = []
        if self.tfidf:
            combination.append(self.tf(data, istrain))
        elif self.countvec:
            combination.append(self.countv(data, istrain))
        if self.number_of_words:
            combination.append(self.numofwords(data))
        if self.number_of_exclamatory:
            combination.append(self.numofexclam(odata))
        if self.number_of_numbers:
            combination.append(self.numofnum(odata))

        matrix = [[] for _ in range(len(data))]
        for k in range(len(combination)):
            for i in range(len(combination[k])):
                for j in combination[k][i]:
                    matrix[i].append(j)
        self.generate(matrix, labels, output_filename)

    def generate(self, matrix, labels, output_filename):
        with open(output_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            fieldnames = ['id'] + [('d' + str(i)) for i in range(1, len(matrix[0]) + 1)] + ['label']
            writer.writerow(fieldnames)
            for i, row in enumerate(matrix):
                new_row = [i]
                for r in row:
                    new_row.append(r)
                new_row.append(labels[i])
                writer.writerow(new_row)

    def read_data(self, filename, col1, col2, header=True):
        if not header:
            raw_data = pd.read_csv(filename, header=None)
        else:
            raw_data = pd.read_csv(filename)
        data, labels = raw_data[col1].tolist(), raw_data[col2].tolist()
        return data, labels

    def tf(self, data, istrain):
        if istrain:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(data)
            tfidf_matrix = tfidf_matrix.toarray()
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(data)
            tfidf_matrix = tfidf_matrix.toarray()
        return tfidf_matrix

    def countv(self, data, istrain):
        if istrain:
            count_matrix = self.count_vectorizer.fit_transform(data)
            count_matrix = count_matrix.toarray()
        else:
            count_matrix = self.count_vectorizer.transform(data)
            count_matrix = count_matrix.toarray()
        return count_matrix

    def numofwords(self, data):
        res = [[len(data[i].split())] for i in range(len(data))]
        return res

    def numofexclam(self, data):
        res = [[0] for i in range(len(data))]
        for i in range(len(data)):
            for ch in data[i]:
                if ch == '!':
                    res[i][0] += 1
        return res

    def numofnum(self, data):
        res = [[0] for i in range(len(data))]
        for i in range(len(data)):
            for ch in data[i]:
                if ch.isdigit():
                    res[i][0] += 1
        return res


fe = FeatureEngineer(tfidf=True, number_of_words=True, number_of_exclamatory=True, number_of_numbers=True)
# fe = FeatureEngineer(tfidf=False,countvec=True,number_of_words=True, number_of_exclamatory=True)
fe.encode(process_input_filename='process_train.csv', original_input_filename='./dataset/raw_data/fulltrain.csv',
          output_filename='feature_train.csv')
fe.encode(process_input_filename='process_test.csv', original_input_filename='./dataset/raw_data/balancedtest.csv',
          output_filename='feature_test.csv', istrain=False)