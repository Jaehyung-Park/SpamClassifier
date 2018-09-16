import pickle
import os
from sklearn import *
from collections import Counter


def load(clf_file):
    with open(clf_file) as fp:
        clf = pickle.load(fp)
    return clf

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

clf = load("text-classifier.mdl")
d = make_dict()

while True:
    feaures = []
    inp = raw_input(">").split()
    if inp[0]  == "exit":
        break
    for word in d:
        features.append(inp.count(word[0]))
    res = clf.predict([features])
    print(["Not Spam","Spam!"][res[0]])
