import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
df=pd.read_csv("Twitter_Spams.csv")
df.drop(columns=["sn"],inplace=True)
x=df['FORMATTED_CONTENT']
y=df["CLASS"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
vectrizer=TfidfVectorizer()
x_train_tfidf=vectrizer.fit_transform(x_train)
x_test_tfidf=vectrizer.transform(x_test)
model=LogisticRegression()
model.fit(x_train_tfidf,y_train)
y_predict=model.predict(x_test_tfidf)
report=classification_report(y_test,y_predict)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predict)

print(cm)

