import numpy as np 
import pandas as pd 
from plotly import express
from wordcloud import WordCloud
import matplotlib.pyplot as plt

data = pd.read_csv('Reddit_Combi.csv', sep=';')

data.head()

data.shape

data.info

data.describe().T

data['label'].value_counts()

print(data.isnull().sum())

data.duplicated().sum()


#Data Visualization

express.pie(data_frame=data.copy()['label'].replace({0: 'negative', 1: 'positive'}),
            names='label', color='label')


def generate_word_cloud(text, title):
    wordcloud = WordCloud(width = 800, height = 400, background_color ='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(title)
    plt.show()

# Concatenating all Body_Title for each label
text_0 = " ".join(data[data['label'] == 0]['Body_Title'])
text_1 = " ".join(data[data['label'] == 1]['Body_Title'])

# Generate word clouds for each label
generate_word_cloud(text_0, "Word Cloud for Label 0")
generate_word_cloud(text_1, "Word Cloud for Label 1")


import nltk
nltk.download('vader_lexicon')


import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer


# Initializing the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Applying sentiment analysis
data['sentiment_scores'] = data['Body_Title'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Plotting sentiment scores distribution for each label
plt.figure(figsize=(10, 6))
sns.histplot(data, x='sentiment_scores', hue='label', element="step", stat="density", common_norm=False)
plt.title('Distribution of Sentiment Scores by Label')
plt.xlabel('Sentiment Score')
plt.ylabel('Density')
plt.show()



# logistic regression


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test= train_test_split(data['Body_Title'], data['label'], test_size=0.2, random_state=42 )


# Initialize a TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# Fit and transform the training data, and transform the testing data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)
lr_accuracy = accuracy_score(y_test, lr_pred)

# SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_tfidf, y_train)
svm_pred = svm_model.predict(X_test_tfidf)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
rf_pred = rf_model.predict(X_test_tfidf)
rf_accuracy = accuracy_score(y_test, rf_pred)

# XGBClassifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_tfidf, y_train)
xgb_pred = xgb_model.predict(X_test_tfidf)
xgb_accuracy = accuracy_score(y_test, xgb_pred)


# Creating a DataFrame for accuracies
accuracies = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'Random Forest', 'XGBClassifier'],
    'Accuracy': [lr_accuracy, svm_accuracy, rf_accuracy, xgb_accuracy]
})

print(accuracies)


from sklearn.model_selection import cross_val_score

svm_model = SVC(kernel='linear')
scores = cross_val_score(svm_model, X_train_tfidf, y_train, cv=5)


print("Average cross-validated accuracy of SVM model:", scores.mean())


from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validated score:", grid_search.best_score_)


