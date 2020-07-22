from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer                 
import datetime                 
import numpy as np                  
import matplotlib.pyplot as plt                 

from sklearn.model_selection import train_test_split
from sklearn import datasets
def read_from_file(file_name, is_label_file=False):                                           # function that reads data from a file and processes it
    f = open(file_name, 'r')                                        
    data = f.read()                                                                           # reading the data
    lines = data.split('\n')                                                                  # splitting the data into lines
    first, second = [], []                                                                                
    for i in range(len(lines) - 1):                                     
        split = lines[i].split('\t')                                                          # splitting the data by '\t'
        first.append(int(split[0]))                                                           # left side of '\t'
        if (is_label_file == True):                                                           # checking if the function reads a label file so it can convert the class to int
            second.append(int(split[1]))                                        
        else:                           
            split[1] = split[1][::-1]                               
            second.append(split[1])                                                               
    return first, second                                            


def normalize_data(train_data, test_data, type=None):                                         # function that normalizes the data using different normalizing methods
    if (type == 'l1' or type == 'l2'):
        normalizer = Normalizer(norm=type)
        norm_train_data = normalizer.transform(train_data)
        norm_test_data = normalizer.transform(test_data)
        return norm_train_data, norm_test_data

    

def accuracy_score(test_samples, test_predictions):
    accepted = 0
    for i in range(len(test_predictions)):
        if test_samples[i] == test_predictions[i]:
            accepted += 1
    accepted /= len(test_predictions)
    return accepted

def bag_of_words(train_data, test_data):
    train_cv = CountVectorizer(token_pattern=r"\w\w[A-Za-z0-9!@#$%&*(}:;',)]+")
    bow_train_data = train_cv.fit_transform(train_data).toarray()
    print(len(train_cv.get_feature_names()))
    bow_test_data = train_cv.transform(test_data).toarray()
    return bow_train_data, bow_test_data

def support_vector_machine(train_data, train_labels, test_data):
    classifier = SVC(gamma='scale', C=1, kernel='rbf')
    print(len(train_data), len(test_data))
    classifier.fit(train_data, train_labels)
    currentDT = datetime.datetime.now()
    print (str(currentDT), " - TRAINING FINISHED")
    predictions = classifier.predict(test_data)
    currentDT = datetime.datetime.now()
    print (str(currentDT), " - TESTING FINISHED")
    return predictions

def write_submission(validation_index, predictions, file_name):
    f = open(file_name, 'w')
    f.write("id,label\n")
    for i in range (len(validation_index)):
        f.write(str(validation_index[i])+','+str(predictions[i])+'\n')



currentDT = datetime.datetime.now()
print (str(currentDT), " - PROGRAM STARTED")
train_index, train_data = read_from_file('train_samples.txt')   
train_index, train_labels  = read_from_file('train_labels.txt', True)
test_index, test_data = read_from_file('test_samples.txt')
validation_index, validation_data = read_from_file('validation_samples.txt')
validation_index, validation_labels = read_from_file('validation_labels.txt', True)
train_data += validation_data
train_labels += validation_labels
bow_train_data, bow_test_data = bag_of_words(train_data, test_data)
norm_train_data, norm_test_data = normalize_data(bow_train_data, bow_test_data, 'l2')
predictions = support_vector_machine(norm_train_data, train_labels, norm_test_data)
#print(accuracy_score(validation_labels, predictions))
write_submission(test_index, predictions, "predictions.txt")

