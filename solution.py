"""
COMP9417 20T2

Name: Mukul Raj Sharma
zID: z5220980
Topic: Sentiment Analysis On Movie Reviews (Kaggle Competition)

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from matplotlib import pyplot as plot
from time import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks


# downloads
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# load the training and testing data from tsv files
train = pd.read_csv("train.tsv", sep="\t")
test = pd.read_csv("test.tsv", sep="\t")

# get the basic stucture of the dataset
train.head()
train.shape

test.head()
test.shape

# plot a pie chart to visualize the distribution of data among classes (percentage)
plot_size = plt.rcParams["figure.figsize"]
plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size

train.Sentiment.value_counts().plot(kind="pie", autopct="%1.0f%%")

# plot a pie chart to visualize the distribution of data among classes (count)
fig = plot.figure(figsize=(15, 5))
sns.countplot(data=train, x='Sentiment')
plot.show()


# clean the data by removing not so useful characters
# these include stopwords, alphanumeric chars and punctuations

def clean(cols):
    clean_ft = []
    for i in range(0, len(cols)):
        # remove stopwords
        ft = remove_stops(cols[i])
        # remove all alphanumerics
        ft = re.sub(r'\W', ' ', str(cols[i]))
        # remove whitespaces
        ft = re.sub(r'\s+', ' ', ft, flags=re.I)
        # convert all chars to lowercase
        ft = ft.lower()
        clean_ft.append(ft)
    return clean_ft

# remove the stop words listed in the nltk english library

stop_words = set(stopwords.words('english'))
def remove_stops(review):    
    tokenized = word_tokenize(review)
    filtered_words= [word for word in tokenized if not word in stop_words]
    return ' '.join(filtered_words)

# extract the feature column (phrases) and the labels (Sentiments)
features = train.iloc[:, 2].values
labels = train.iloc[:, 3].values

# call the clean funtion on the above data
start_time = time()
clean_phrases = clean(features)
end_time = time()
spent = end_time-start_time
print("time to clean training data: {}".format(spent))

# attach clean phrases to the trainset
train['cleanPhrase'] = clean_phrases
train.head()

# make a stratified split for train and test data (66/33 split)
X_train, X_test, y_train, y_test = train_test_split(train['cleanPhrase'], labels, test_size=0.33, random_state=42, stratify=labels)


# vectorize the text data in X_train, y_train to make it compatible with the
# classification model
vectorizer = TfidfVectorizer(min_df=3,  max_features=None, ngram_range=(1, 2), use_idf=1)


start = time()
vec_X_train = vectorizer.fit_transform(X_train)
vec_X_test = vectorizer.transform(X_test)
end = time()
print("time taken to vecotrize {}".format(end-start))

# sanity check on how the vectorized features look
vec_X_train[0]



"""

Check the performance of the clean data on different classification models:

1. Logistic regressions
2. Multinomial Naive Bayes
3. Random forrest Classifier

"""


# Logistic Regression

print("Using Logistic Regression : \n")
model_lr = LogisticRegression()
model_lr.fit(vec_X_train, y_train)
vec_X_test = vectorizer.transform(X_test)
predicted = model_lr.predict(vec_X_test)
expected = y_test

cm = metrics.confusion_matrix(expected,predicted)
print(metrics.classification_report(expected, predicted))
print("Accuracy in Cross-Validation Set : ", metrics.accuracy_score(expected, predicted))

# return an accuracy of 64%
# Visualize the performance on LR on different classes
sns.heatmap(cm, annot=True,cmap='OrRd')
plt.show()

# Multinomial Naive Bayes

print("Using Multinomial Naive Bayes : \n")
model_mnb = MultinomialNB()
model_mnb.fit(vec_X_train, y_train)
vec_X_test = vectorizer.transform(X_test)
predicted = model_mnb.predict(vec_X_test)
expected = y_test

cm = metrics.confusion_matrix(expected,predicted)
print(metrics.classification_report(expected, predicted))
print("Accuracy in Cross-Validation Set : ", metrics.accuracy_score(expected, predicted))

# return an accuracy of 61%
# Visualize the performance on MNB on different classes

sns.heatmap(cm, annot=True,cmap='OrRd')
plt.show()


# Random Forest Classifier
# the result changes by 1-2% on decresing the estimators to 100

print("Using Random Forest Classifier: \n")
start_time = time()
model_rfc = RandomForestClassifier(max_depth = None, n_estimator = 300, n_jobs=-1, verbose=3)
model_rfc.fit(vec_X_train, y_train)
vec_X_test = vectorizer.transform(X_test)
predicted = model_rfc.predict(vec_X_test)
expected = y_test
end_time = time()
print("time taken by random forest: {}".format(end_time-start_time))

cm = metrics.confusion_matrix(expected,predicted)
print(metrics.classification_report(expected, predicted))
print("Accuracy in Cross-Validation Set : ", metrics.accuracy_score(expected, predicted))

# return an accuracy of 66%
# Visualize the performance on RFC on different classes

sns.heatmap(cm, annot=True,cmap='OrRd')
plt.show()


"""
Since we observed that our data has an imbalance in sample distribution, 
we will apply 2 techniques to try mitigate that.

The first tecnique used is Under Sampling.
The second one is Tomek Links.

"""

# For under sampling I will be using RandomUnderSampler

rus = RandomUnderSampler(random_state=0)
x_rus, y_rus = rus.fit_sample(vec_X_train, y_train)

print(x_rus.shape)
print(y_rus.shape)
print(x_rus[0])

# create the under sampled dataset
rus_data = {'x_val':x_rus, 'sentiment':y_rus}
rus_df = pd.DataFrame(rus_data, columns=['x_val','sentiment'])

# visulize the new equal distribution of classes
fig = plot.figure(figsize=(15, 5))
sns.countplot(data=rus_df, x='sentiment')
plot.show()
rus_df.sentiment.value_counts().plot(kind="pie", autopct="%1.0f%%")


# Create the new dataset with tomek links
tomekl = TomekLinks(random_state=0,n_jobs=3)

x_tomekl, y_tomekl = tomekl.fit_sample(vec_X_train, y_train)

tomekl_data = {'x_val':x_tomekl, 'sentiment':y_tomekl}
tomekl_df = pd.DataFrame(tomekl_data, columns=['x_val','sentiment'])

tomekl_df.sentiment.value_counts().plot(kind="pie", autopct="%1.0f%%")

# run LR with the new tomekl data
print("Using Logistic Regression on rus data: \n")
model = LogisticRegression()
model.fit(x_rus, y_rus)
vec_X_test = vectorizer.transform(X_test)
predicted = model.predict(vec_X_test)
expected = y_test

# visualize the effects of tomekl links on the dataset
cm = metrics.confusion_matrix(expected,predicted)
print(metrics.classification_report(expected, predicted))
print("Accuracy in Cross-Validation Set : ", metrics.accuracy_score(expected, predicted))

sns.heatmap(cm, annot=True,cmap='OrRd')
plt.show()

# sanity check on the data after using tomekl links
print(x_tomekl.shape)
print(y_tomekl.shape)
print(x_tomekl[0])
print(y_tomekl[0])

sns.heatmap(cm, annot=True,cmap='OrRd')
plt.show()

"""

I ran the RUS and Tomekl models on all the models that we saw before, 
the inference can be found in the report.

"""
'''
The code below was an attempt to implement the LSTM model however the training
time was too long and the accuracy was ~55% so it seemed like I was doing something
incorrectly. Hence, I decided to work on this in the future.

'''

# model = Sequential()
# model.add(Embedding(14219,300))
# model.add(LSTM(150))
# model.add(Dense(5,activation='softmax'))
# model.summary()

# model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# history=model.fit(vec_X_train, y_train, validation_data=(vec_X_test, y_test),epochs=6, batch_size=256, verbose=1)