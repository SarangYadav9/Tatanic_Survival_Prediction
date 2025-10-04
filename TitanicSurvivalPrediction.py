print("-----------------------------------------Titanic Survival Prediction-------------------------------------------------")
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/dexter/Documents/python/Titanic.csv')
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'female' :1, 'male':0})

model = LogisticRegression()

features = ['Age','Sex','Pclass']
x = df[features]
y = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
model.fit(x_train,y_train)

new_age = float(input("Enter age of Passanger :"))
sex = int(input('Enter Gender of Passanger(female-1,male-0) :'))
new_pclass = int(input("Enter Pclass of Passanger(1,2 or 3) :"))
df1 = pd.DataFrame([[new_age,sex,new_pclass]],columns=features)
pred = model.predict(df1)

if pred[0]==1:
    print("The Passanger is Survived!")
else:
    print("The Passanger Not Survived!")