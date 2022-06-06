from __future__ import print_function
import numpy as np
from sklearn.metrics import accuracy_score

def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(solver='liblinear', multi_class='ovr')
    log.fit(train_embeds, train_labels)
    predict = (log.predict(test_embeds)).tolist()
    accuracy = accuracy_score(test_labels, predict)
    print("Test Accuracy:", accuracy)
    return accuracy, predict

def classify(embeds, dataset, per_class):

    label_file = open("data/{}{}".format(dataset,"_labels.txt"), 'r')
    label_text = label_file.readlines()
    labels = []
    for line in label_text:
        if line.strip('\n'):
            line = line.strip('\n').split(' ')
            labels.append(int(line[1]))
    label_file.close()
    labels = np.array(labels)
    train_file = open("data/{}/{}/train_text.txt".format(dataset, per_class), 'r')
    train_text = train_file.readlines()
    train_file.close()
    test_file = open( "data/{}/{}/test_text.txt".format(dataset, per_class), 'r')
    test_text = test_file.readlines()
    test_file.close()
    ave = []
    for k in range(50):
        train_ids = eval(train_text[k])
        test_ids = eval(test_text[k])
        train_labels = [labels[i] for i in train_ids]
        test_labels = [labels[i] for i in test_ids]
        train_embeds = embeds[[id for id in train_ids]]
        test_embeds = embeds[[id for id in test_ids]]
       # print(test_labels)
        acc, _ = run_regression(train_embeds, train_labels, test_embeds, test_labels)
        ave.append(acc)
    print(np.mean(ave)*100)
    print(np.std(ave)*100)

def classify_ours(embeds, labels, idx_train, idx_test):

    labels = np.array(labels)
    ave = []
    train_labels = [labels[i] for i in idx_train]
    test_labels = [labels[i] for i in idx_test]
    train_embeds = embeds[[id for id in idx_train]]
    test_embeds = embeds[[id for id in idx_test]]
    # print(test_labels)
    acc, pred = run_regression(train_embeds, train_labels, test_embeds, test_labels)
    return acc, pred

