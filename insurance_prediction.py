import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data=pd.read_csv('insurance_data.csv')
X=data[['age']]
y=data['bought_insurance']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
print('Model Accuracy:',round(model.score(X_test,y_test)*100,2),'%')
age=int(input('Enter Age: '))
pred=model.predict([[age]])
print('Bought Insurance: Yes (1)' if pred[0]==1 else 'Bought Insurance: No (0)')
