# Ex.No: 13 Learning – Use Supervised Learning  
### DATE: 22/04/24                                                                      
### REGISTER NUMBER : 212221040145
### AIM: 
To write a program to train the classifier for Diabetes.
###  Algorithm:

Step 1: Import packages 

Step 2: Get the data 

Step 3: Split the data 

Step 4: Scale the data

Step 5: Instantiate model 

Step 6: Create a function for gradio 

Step 7: Print Result.
### Program:

```py
import numpy as np
import pandas as pd
pip install gradio
pip install typing-extensions --upgrade
pip install --upgrade typing
```

```py
import gradio as gr
import pandas as pd
```

```py
data = pd.read_csv('diabetes.csv')
data.head()
```

```py
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
print(x[:5])
```

```py
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y)
```

```py
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
```

```py
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train, y_train)
print("Model Accuracy on training set:", model.score(x_train, y_train))
print("Model Accuracy on Test Set:", model.score(x_test, y_test))
print(data.columns)
```

```py
def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin, BMI,Diabetes_Pedigree, Age):
    x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])
    prediction = model.predict(x.reshape(1, -1))
    if(prediction==0):
      return "NO"
    else:
      return "YES"

```

```py
outputs = gr.Textbox()
app = gr.Interface(fn=diabetes, inputs=['number','number','number','number','number','number','number','number'], outputs=outputs,description="Detection of Diabeties")
app.launch(share=True)
```
### Output:

1.Dataset

![WhatsApp Image 2024-04-25 at 09 23 58_77b41ef5](https://github.com/kannan0071/AI_Lab_2023-24/assets/119641638/902fe083-c720-44c7-9118-d38f78aed76e)

2.Accuracy

![WhatsApp Image 2024-04-25 at 09 24 12_1d8641b3](https://github.com/kannan0071/AI_Lab_2023-24/assets/119641638/c3831e9b-d4ff-4f4a-877c-df8603faed09)

3.Result

![WhatsApp Image 2024-04-25 at 09 23 34_e1841a5d](https://github.com/kannan0071/AI_Lab_2023-24/assets/119641638/d376bb90-777a-4780-87e7-51168b7111a2)

### Result:
Thus the system was trained successfully and the prediction was carried out.
