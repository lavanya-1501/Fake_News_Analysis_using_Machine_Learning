import itertools

from django.db.models import Q
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine, true

from tkinter import *
from tkinter import ttk, filedialog
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy



from django.shortcuts import render, redirect

# Create your views here.
from client.forms import RegisterForms
from client.models import RegisterModel
from management.models import AdminModel



def login(request):
    if request.method=="POST":
        usid=request.POST.get('username')
        pswd = request.POST.get('password')
        try:
            check = RegisterModel.objects.get(userid=usid,password=pswd)
            request.session['userd_id']=check.id
            return redirect('upload_news')
        except:
            pass
    return render(request,'client/login.html')

def register(request):
    if request.method == "POST":
        forms = RegisterForms(request.POST)
        if forms.is_valid():
            forms.save()

            return redirect('register')
    else:
        forms = RegisterForms()
    return render(request, 'client/register.html',{'form':forms})


def mydetails(request):
    name = request.session['userd_id']
    obj = RegisterModel.objects.get(id=name)
    return render(request, 'client/mydetails.html', {'objects': obj})


def analysis(request):
    return render(request, 'client/analysis.html')


def tfidf(request):
    engine = sqlalchemy.create_engine("mysql+pymysql://root:@localhost/fake_news", pool_pre_ping=True)
    df = pd.read_sql_table('management_adminmodel', engine)

    y = df.label
    df.drop("label", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.3, random_state=None)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    clf = MultinomialNB()

    clf.fit(tfidf_train, y_train)
    pred = clf.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    score1 = round(score * 100, 2)
    print("Accuracy:   %0.3f" % score)
    print(f'Accuracy: {round(score * 100, 2)}%')

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Multinomial Naive Bayes using TFIDF Vectorizer - ' + str(score1) + '%',
                              cmap=plt.cm.Blues):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        figure = plt.gcf()
        figure.set_size_inches(32, 18)

        plt.show()
        plt.close(figure)
        figure.clear()

    cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
    return render(request, 'client/tfidf.html')


def count(request):
    engine = sqlalchemy.create_engine("mysql+pymysql://root:@localhost/fake_news", pool_pre_ping=True)
    df = pd.read_sql_table('management_adminmodel', engine)

    y = df.label

    # Drop the `label` column
    df.drop("label", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.3, random_state=None)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test set
    count_test = count_vectorizer.transform(X_test)

    clf = MultinomialNB()

    clf.fit(count_train, y_train)
    pred = clf.predict(count_test)
    score = metrics.accuracy_score(y_test, pred)
    print("Accuracy:   %0.3f" % score)
    score1 = round(score * 100, 2)

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Multinomial Naive Bayes Classifier using Count Vectorizer - ' + str(score1) + '%',
                              cmap=plt.cm.Greens):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        plt.show()

    cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

    return render(request, 'client/count.html')


def passiveaggressive(request):
    engine = sqlalchemy.create_engine("mysql+pymysql://root:@localhost/fake_news", pool_pre_ping=True)
    df = pd.read_sql_table('management_adminmodel', engine)

    # Inspect shape of `df`
    df.shape

    # Set `y`
    y = df.label

    # Drop the `label` column
    df.drop("label", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=None)

    # Initialize the `tfidf_vectorizer`
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test set
    tfidf_test = tfidf_vectorizer.transform(X_test)

    linear_clf = PassiveAggressiveClassifier()

    linear_clf.fit(tfidf_train, y_train)
    pred = linear_clf.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    score1 = round(score * 100, 2)
    print("Accuracy:   %0.3f" % score)

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Passive Aggressive Classifier - ' + str(score1) + '%',
                              cmap=plt.cm.Oranges):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        plt.show()
    cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
    return render(request, 'client/passiveaggressive.html')


def hashing(request):
    engine = sqlalchemy.create_engine("mysql+pymysql://root:@localhost/fake_news", pool_pre_ping=True)
    df = pd.read_sql_table('management_adminmodel', engine)

    # Set `y`
    y = df.label

    # Drop the `label` column
    df.drop("label", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=None)

    hash_vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False)
    hash_train = hash_vectorizer.fit_transform(X_train)
    hash_test = hash_vectorizer.transform(X_test)

    clf = MultinomialNB(alpha=.01)

    clf.fit(hash_train, y_train)
    pred = clf.predict(hash_test)
    score = metrics.accuracy_score(y_test, pred)
    score1 = round(score * 100, 2)
    print("Accuracy:   %0.3f" % score)

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Multinomial Naive Bayes using Hashing Vectorizer - ' + str(score1) + '%',
                              cmap=plt.cm.Purples):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        plt.show()

    cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
    return render(request, 'client/hashing.html')



def topreal(request):
    engine = sqlalchemy.create_engine("mysql+pymysql://root:@localhost/fake_news", pool_pre_ping=True)
    df = pd.read_sql_table('management_adminmodel', engine)
    real_data = df[df["label"] == "REAL"]
    all_words = ' '.join([text for text in real_data.text])
    wordcloud = WordCloud(width=800, height=500,
                          max_font_size=110,
                          collocations =False).generate(all_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    return render(request, 'client/topreal.html')

def topfake(request):
    engine = sqlalchemy.create_engine("mysql+pymysql://root:@localhost/fake_news", pool_pre_ping=True)
    df = pd.read_sql_table('management_adminmodel', engine)
    fake_data = df[df["label"] == "FAKE"]
    all_words = ' '.join([text for text in fake_data.text])
    wordcloud = WordCloud(width=800, height=500,
                          max_font_size=110,
                          collocations =False).generate(all_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    return render(request, 'client/topfake.html')



def upload_news(request):
    if request.method == "POST":
        newsid =request.POST.get('newsid')
        title = request.POST.get('title')
        text = request.POST.get('text')
        label = request.POST.get('label')

        AdminModel.objects.create(newsid=newsid, title=title, text=text, label=label)
    return render(request,"client/upload_news.html")



def upload_dataset(request):
    win = Tk()
    win.geometry("700x350")

    def open_file():
        global df
        import_file_path = filedialog.askopenfilename()
        df = pd.read_csv(import_file_path)
        sqlalchemy.String(1000, convert_unicode=True)
        print(df)
        engine = create_engine('mysql+pymysql://root:@localhost/fake_news')
        with engine.connect() as conn, conn.begin():
            df.to_sql('management_adminmodel', conn, if_exists='append', index=False)

    label = Label(win, text="Click the Button to browse the Files", font=('Georgia 13'))
    label.pack(pady=10)
    ttk.Button(win, text="Browse", command=open_file).pack(pady=20)

    win.mainloop()
    return render(request,'client/upload_dataset.html')


def view_upload(request):
    obj1 = AdminModel.objects.all()
    return render(request,'client/view_upload.html',{'obj1':obj1})

