import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import pickle

def save(clf, name):
    with open(name, 'wb') as fp:
        pickle.dump(clf, fp)
    print ("saved")

def make_dict():
    direc = "emails/"
    files = os.listdir(direc)

    emails = [direc +email for email in files]

    words = []
    c = len(emails)
    for email in emails:
        f=open(email,encoding="latin-1")
        blob = f.read()
        words += blob.split(" ")
        print(c)
        c -= 1

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""

    dictionary = Counter(words)
    del dictionary[""]
    return (dictionary.most_common(3000))

#making all the emails of the text files into feature vectors

def make_dataset(dictionary):
    #reading the email once again
    direc = "emails/"
    files = os.listdir(direc)

    emails = [direc +email for email in files]

    feature_set = []
    labels = []
    c = len(emails)
    for email in emails:
        data = []
        f=open(email,encoding="latin-1")
        words = f.read().split(' ')
        for entry in dictionary:
            data.append(words.count(entry[0]))
            #since entry is a dictioary element and has both a key and a value pair.
            #Key is the word and the value is the frequency. we are only interested in words
        feature_set.append(data)
        if "ham" in email:
            labels.append(0)
        if "spam" in email:
            labels.append(1)
        print (c)
        c = c-1
    return feature_set, labels

d = make_dict()
features, labels = make_dataset(d)

x_train, x_test, y_train, y_test =tts(features,labels,test_size = 0.2)
#80 percent for data for training 20percent is kept aside
clf = MultinomialNB()
clf.fit(x_train,y_train)

preds = clf.predict(x_test)
print (accuracy_score(y_test,preds))
save(clf,"text-classifer.mdl")
