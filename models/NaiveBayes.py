import pandas as pd
import re
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
import time


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)


def predict(model, X_test):
    return model.predict(X_test)


def main():
    ''' load train, val, and test data '''
    time0 = time.time()
    train = pd.read_csv('./dataset/raw_data/fulltrain.csv',names=['label','text'])
    # train = pd.read_csv('./process_train.csv')
    # train = pd.read_csv('./feature_train.csv')
    # X_train = train.drop(['id', 'label'], axis=1)
    X_train = train['text']
    y_train = train['label']
    # feature engineering
    tfidf = TFIDF().fit(X_train)
    X_train = tfidf.transform(X_train)

    # define the model
    # candidate_models: MultinomialNB, ComplementNB, BernoulliNB              # Train,Test
    # model = MultinomialNB()                                                 # 0.592,0.319
    # model = CalibratedClassifierCV(MultinomialNB(), cv=2, method='isotonic')# 0.791,0.550
    # model = CalibratedClassifierCV(MultinomialNB(), cv=2, method='sigmoid') # 0.675,0.408
    # model = ComplementNB()                                                  # 0.839,0.484
    # model = CalibratedClassifierCV(ComplementNB(), cv=2, method='isotonic') # 0.858,0.580
    # model = CalibratedClassifierCV(ComplementNB(), cv=2, method='sigmoid')  # 0.850.0.567
    # model = BernoulliNB()                                                   # 0.778,0.596
    # model = CalibratedClassifierCV(BernoulliNB(), cv=2, method='isotonic')  # 0.786,0.593
    model = CalibratedClassifierCV(BernoulliNB(), cv=2, method='sigmoid')     # 0.772,0.607

    train_model(model, X_train, y_train)
    y_pred = predict(model, X_train)

    # Use f1-macro as the metric
    score = f1_score(y_train, y_pred, average='macro')
    print('Train F1 score = {}'.format(score))

    # generate prediction on test data
    test = pd.read_csv('./dataset/raw_data/balancedtest.csv',names=['label','text'])
    # test = pd.read_csv('./process_test.csv')
    # test = pd.read_csv('./feature_test.csv')
    # X_test = test.drop(['id', 'label'], axis=1)
    X_test = test['text']
    y_test = test['label']
    X_test = tfidf.transform(X_test)
    y_pred = predict(model, X_test)
    score = f1_score(y_test, y_pred, average='macro')
    print('Test F1 score = {}'.format(score))
    print(time.time() - time0, 's')


# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()