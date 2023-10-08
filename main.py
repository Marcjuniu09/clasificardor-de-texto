import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os


#Aqui preferi deixar os dois modelos juntos, mas poderia ter feito uma função para cada modelo. Achei que ficou melhor assim
def train(df, model):
    """
    df: dataframe
    model: modelo a ser treinado (KNN ou Naive Bayes)
    return: accuracy
    """
    x_train, x_test, y_train, y_test = train_test_split(df['Text'], df['Class'], test_size=0.2, random_state=5)
    tfidf_vec = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vec.fit_transform(x_train)
    tfidf_test = tfidf_vec.transform(x_test)
    if model == 'naive_bayes':
        clf = MultinomialNB()
        clf.fit(tfidf_train, y_train)
        pred = clf.predict(tfidf_test)

    elif model == 'knn':
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(tfidf_train, y_train)
        pred = knn.predict(tfidf_test)

    score = accuracy_score(y_test, pred)
        
    return ("accuracy:   %0.3f" % score)


def main():
    files =  [file for file in os.listdir('data/') if file.endswith('.csv')]

    df = pd.concat([pd.read_csv('data/' + file) for file in files])
    list_models = ['naive_bayes', 'knn']
    df = pd.read_parquet('data/df.parquet')
    for model in list_models:
        print(model)
        print(train(df, model))
        print('-------------------------')
        
        
if __name__ == '__main__':  
    main()