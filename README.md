# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect and load the dataset containing city features (independent variable) and profit (dependent variable).
2. Initialize model parameters (weights and bias) with initial values and set the learning rate.
3. Compute the predicted profit, calculate the error, and update parameters using gradient descent repeatedly.
4. Stop after a fixed number of iterations or when the error is minimal, and obtain the final regression model.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Denesh Raj Balaji Rao
RegisterNumber: 212225230047
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

```
data = pd.read_csv("50_Startups.csv")
x = data["R&D Spend"]. values
y = data["Profit"].values
```

```
x_mean = np.mean(x)
x_std = np.std(x)
x = (x - x_mean) / x_std
```

```
w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)
losses = []
```

```
for i in range(epochs):
    y_hat = w * x + b
    
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)
    
    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db
```

```
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1, 2, 2)
plt.scatter(x, y)
x_sorted = np.argsort(x)
plt.plot(x[x_sorted], (w * x + b)[x_sorted], color='red')
plt.xlabel("R&D Spend (Scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")

plt.tight_layout()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)
```


## Output:
<img width="1040" height="457" alt="image" src="https://github.com/user-attachments/assets/95fa1ed1-545b-4997-8f77-66ee7a950017" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
