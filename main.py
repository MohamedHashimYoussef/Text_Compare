from wordcloud import WordCloud
import matplotlib.pyplot as plt
# 1-load dataset
import pandas as pd
import joblib
import pickle, joblib
df = pd.read_csv('train.csv')


# df.head(5) # for showing a snapshot of the dataset
# 2-Text Processing Steps
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatiser = WordNetLemmatizer()


# Defining a module for Text Processing
def text_process(tex):
   # print(tex)
    # 1. Removal of Punctuation Marks
    nopunct = [char for char in tex if char not in string.punctuation]
    nopunct = ''.join(nopunct)
    # 2. Lemmatisation
    a = ''
    i = 0
    for i in range(len(nopunct.split())):
        b = lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        a = a + b + ' '
    # 3. Removal of Stopwords
    return [word for word in a.split() if word.lower() not
            in stopwords.words('english')]


# 3-Label Encoding of Classes
from sklearn.preprocessing import LabelEncoder

y = df['author'].to_numpy()
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
print(y)
# 4-Word Cloud Visualization
from PIL import Image


X = df['text'].to_numpy()
#print(X)

wordcloud1 = WordCloud().generate(X[0])  # for EAP
wordcloud2 = WordCloud().generate(X[1])  # for HPL
wordcloud3 = WordCloud().generate(X[3])  # for MWS print(X[0])
#print(df['author'][0])
#plt.imshow(wordcloud1, interpolation='bilinear')
#plt.show()
#print(X[1])
#print(df['author'][1])
#plt.imshow(wordcloud2, interpolation='bilinear')
#plt.show()
#print(X[3])
#print(df['author'][3])
#plt.imshow(wordcloud3, interpolation='bilinear')
#plt.show()
# 4-Feature Engineering using Bag-of-Words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  # 80-20 splitting the dataset (80%->Training and 20%->Validation)

X_train, X_test, y_train, y_test = train_test_split(X, y
                                                    , test_size=0.2,
                                                    random_state=1234)
#print(X_test[0])
#print(X_test[0])
 #defining the bag-of-words transformer on the text-processed corpus # i.e., text_process() declared in II is executed...
bow_transformer = CountVectorizer(analyzer=text_process)
# transforming into Bag-of-Words and hence textual data to numeric..
text_bow_train = bow_transformer.fit_transform(X_train)  # ONLY TRAINING DATA# transforming into Bag-of-Words and hence textual data to numeric..
vectorizer_test = CountVectorizer(vocabulary=bow_transformer.vocabulary_)
#for i in text_bow_train:
#    print(i.shape)
#print('######################')
#print(text_bow_train[0])
#print("this is xtest")
#print(type(X_test))
text_bow_test = bow_transformer.transform(X_test) # TEST DATA
#for i in text_bow_test:
#    print(i.shape)
#print('######################')
#print("text_bow_test test")
#print(type(text_bow_test))
#print(text_bow_test)


# 5-Training the Model
from sklearn.naive_bayes import MultinomialNB  # instantiating the model with Multinomial Naive Bayes..

xx = MultinomialNB()  # training the model...
model = xx.fit(text_bow_train, y_train)

#joblib.dump(model,'saved_model_pkl.pkl')
#exit()
# 6-Model Performance Analysis:
#model.score(text_bow_train, y_train)
#model.score(text_bow_test, y_test)

from sklearn.metrics import classification_report

# getting the predictions of the Validation Set...
#predictions = model.predict(text_bow_test)
# getting the Precision, Recall, F1-Score
#print(classification_report(y_test, predictions))

# Importing necessary libraries
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt  # Defining a module for Confusion Matrix...


#def plot_confusion_matrix(cm, classes,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=plt.cm.Blues):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting normalize=True.
#    """
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#        print(cm)
#        plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0])
#            , range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#        plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#    cm = confusion_matrix(y_test, predictions)


#    plt.figure()
#    plot_confusion_matrix(cm, classes=[0, 1, 2], normalize=True,title='Confusion Matrix')
#model =  joblib.load('saved_model_pkl.pkl')
x=["But a glance will show the fallacy of this idea."]
n=np.array(x)
#vectorizer_train = CountVectorizer(analyzer=text_process)
p= CountVectorizer(vocabulary=bow_transformer.vocabulary_)

bow = p.transform(n)

pr=xx.predict(text_bow_test)
cnt = 0
for i in range(0 , len(y_test)):
    print(y_test[i] , end=' ')
    print(pr[i])
    if y_test[i] == pr[i]:
        cnt+=1
print((float(cnt)/len(y_test))*100)

print(pr)
