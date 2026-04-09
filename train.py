import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

x=[2,4,5,7,9,5,3,6,4]
y=[0,0,0,1,1,0,0,1,0]
X=np.reshape(x, (-1,1))

model = LogisticRegression()
model.fit(X, y)

print("\nLogistic Regression model trained.")
joblib.dump(model,'log.pkl')