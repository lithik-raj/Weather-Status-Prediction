import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv("data.csv")

X = df[['Temperature','Humidity','WindSpeed','Rainfall']]
y = df['WeatherCondition']

model = DecisionTreeClassifier()
model.fit(X,y)

pickle.dump(model,open("model.pkl","wb"))

print("Model Trained Successfully")