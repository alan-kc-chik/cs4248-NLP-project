import os
import csv
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
import gc


class DataLoader:
    def __init__(self, params):
        self.params = params
        self.ntags = params.ntags

        kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}
        # kwargs = {}

        if params.mode == 0:
            self.adj_train = None
            self.train = self.read_dataset_sentence_wise(params.train)

            # Split the training set into two
            self.adj_dev = None
            self.train, self.dev = train_test_split(self.train, test_size=0.2, random_state=42)

            dataset_train = ClassificationGraphDataSet(self.train, self.params)
            self.train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=params.batch_size,
                                                                 collate_fn=dataset_train.collate, shuffle=True,
                                                                 **kwargs)

            dataset_dev = ClassificationGraphDataSet(self.dev, self.params)
            self.dev_data_loader = torch.utils.data.DataLoader(dataset_dev, batch_size=params.batch_size,
                                                               collate_fn=dataset_dev.collate, shuffle=False, **kwargs)
            del self.train
            del self.dev
            gc.collect()

        elif params.mode == 1:
            # Treating this as a binary classification problem for now "1: Satire, 4: Trusted"
            self.adj_test = None
            self.test = self.read_dataset_sentence_wise(params.test)

            dataset_test = ClassificationGraphDataSet(self.test, self.params)
            self.test_data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=params.batch_size,
                                                                collate_fn=dataset_test.collate, shuffle=False,
                                                                **kwargs)

            del self.test
            gc.collect()

    def read_dataset_sentence_wise(self, filename):
        data = []
        with open(filename, "r") as f:
            readCSV = csv.reader(f, delimiter=',')
            csv.field_size_limit(100000000)
            for tag, doc in readCSV:
                sentences = sent_tokenize(doc)
                tag = int(tag)
                allowed_tags = [1, 4] if self.ntags == 2 else [1, 2, 3, 4]
                if tag in allowed_tags:
                    if self.ntags == 2:
                        # Adjust the tag to {0: Satire, 1: Trusted}
                        tag = tag - 1 if tag == 1 else tag - 3
                    else:
                        # {0: Satire, 1: Hoax, 2: Propaganda, 3: Trusted}
                        tag -= 1
                    sentences_filtered = []
                    for sentence in sentences:
                        if len(sentence.strip()) > 0:
                            sentences_filtered.append(sentence)
                    if len(sentences_filtered) > 0:
                        data.append((sentences_filtered, tag))

        return data  # list(tuple(list(sents), tag))


class ClassificationGraphDataSet(torch.utils.data.TensorDataset):
    def __init__(self, data, params):
        super(ClassificationGraphDataSet, self).__init__()
        self.params = params
        # data is a list of tuples (sent, label)
        self.sents = [x[0] for x in data]  # list(list(sents))
        self.labels = [x[1] for x in data]
        self.num_of_samples = len(self.sents)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        return self.sents[idx], len(self.sents[idx]), self.labels[idx]

    def collate(self, batch):
        sents = [x[0] for x in batch]
        doc_lens = np.array([x[1] for x in batch])
        labels = np.array([x[2] for x in batch])
        # Sort sentences within each document by length
        # documents = []

        # max_doc_lens = max(doc_lens)
        # for doc in sents:
        #     padded_doc = doc
        #     if len(doc) < max_doc_lens:
        #         padded_doc.extend([''] * (max_doc_lens - len(doc)))
        #     documents.append(padded_doc)
        # return documents, doc_lens, labels
        return sents, doc_lens, labels



