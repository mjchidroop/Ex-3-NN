<H3>ENTER YOUR NAME</H3> CHIDROOP M J
<H3>ENTER YOUR REGISTER NO.</H3> 212225240029
<H3>EX. NO.3</H3>
<H3>DATE:</H3>
<H2 aligh = center> Implementation of MLP for a non-linearly separable data</H2>
<h3>Aim:</h3>
To implement a perceptron for classification using Python
<H3>Theory:</H3>
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows:

XOR truth table

![Img1](https://user-images.githubusercontent.com/112920679/195774720-35c2ed9d-d484-4485-b608-d809931a28f5.gif)

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below

![Img2](https://user-images.githubusercontent.com/112920679/195774898-b0c5886b-3d58-4377-b52f-73148a3fe54d.gif)

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.To separate the two outputs using linear equation(s), it is required to draw two separate lines as shown in figure below:

![Img 3](https://user-images.githubusercontent.com/112920679/195775012-74683270-561b-4a3a-ac62-cf5ddfcf49ca.gif)

For a problem resembling the outputs of XOR, it was impossible for the machine to set up an equation for good outputs. This is what led to the birth of the concept of hidden layers which are extensively used in Artificial Neural Networks. The solution to the XOR problem lies in multidimensional analysis. We plug in numerous inputs in various layers of interpretation and processing, to generate the optimum outputs.
The inner layers for deeper processing of the inputs are known as hidden layers. The hidden layers are not dependent on any other layers. This architecture is known as Multilayer Perceptron (MLP).

![Img 4](https://user-images.githubusercontent.com/112920679/195775183-1f64fe3d-a60e-4998-b4f5-abce9534689d.gif)

The number of layers in MLP is not fixed and thus can have any number of hidden layers for processing. In the case of MLP, the weights are defined for each hidden layer, which transfers the signal to the next proceeding layer.Using the MLP approach lets us dive into more than two dimensions, which in turn lets us separate the outputs of XOR using multidimensional equations.Each hidden unit invokes an activation function, to range down their output values to 0 or The MLP approach also lies in the class of feed-forward Artificial Neural Network, and thus can only communicate in one direction. MLP solves the XOR problem efficiently by visualizing the data points in multi-dimensions and thus constructing an n-variable equation to fit in the output values using back propagation algorithm

<h3>Algorithm :</H3>

Step 1 : Initialize the input patterns for XOR Gate<BR>
Step 2: Initialize the desired output of the XOR Gate<BR>
Step 3: Initialize the weights for the 2 layer MLP with 2 Hidden neuron  and 1 output neuron<BR>
Step 3: Repeat the  iteration  until the losses become constant and  minimum<BR>
    (i)  Compute the output using forward pass output<BR>
    (ii) Compute the error<BR>
	(iii) Compute the change in weight ‘dw’ by using backward progatation algorithm. <BR>
    (iv) Modify the weight as per delta rule.<BR>
    (v)  Append the losses in a list <BR>
Step 4 : Test for the XOR patterns.

<H3>Program:</H3>

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
y = np.array([[0, 1, 1, 0]])

n_x, n_y, n_h = 2, 1, 2
m = x.shape[1]
lr = 0.1

np.random.seed(2)
w1 = np.random.rand(n_h, n_x)
w2 = np.random.rand(n_y, n_h)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_prop(w1, w2, x):
    z1 = np.dot(w1, x)
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def back_prop(m, w1, w2, z1, a1, z2, a2, y):
    dz2 = a2 - y
    dw2 = np.dot(dz2, a1.T) / m
    dz1 = np.dot(w2.T, dz2) * a1 * (1 - a1)
    dw1 = np.dot(dz1, x.T) / m
    return dz2, dw2, dz1, dw1

iterations = 10000
losses = []

for i in range(iterations):
    z1, a1, z2, a2 = forward_prop(w1, w2, x)
    loss = -(1 / m) * np.sum(y * np.log(a2) + (1 - y) * np.log(1 - a2))
    losses.append(loss)
    dz2, dw2, dz1, dw1 = back_prop(m, w1, w2, z1, a1, z2, a2, y)
    w2 = w2 - lr * dw2
    w1 = w1 - lr * dw1

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()

def predict(w1, w2, input_data):
    z1, a1, z2, a2 = forward_prop(w1, w2, input_data)
    return 1 if np.squeeze(a2) >= 0.5 else 0

print("=" * 50)
print("XOR Gate Predictions")
print("=" * 50)
print("Input  Expected  Predicted  Result")
print("-" * 50)

test_cases = [np.array([[1],[0]]), np.array([[1],[1]]), 
              np.array([[0],[1]]), np.array([[0],[0]])]

for test in test_cases:
    pred = predict(w1, w2, test)
    expected = test[0,0] ^ test[1,0]
    print(f"[{test[0,0]}, {test[1,0]}]     {expected}       {pred}        {'✓' if pred == expected else '✗'}")
print("=" * 50)
```

<H3>Output:</H3>

Training:
```ipynb
Training initialized !!!

Iteration 1000: Loss = 0.6931
Iteration 2000: Loss = 0.6930
Iteration 3000: Loss = 0.6928
Iteration 4000: Loss = 0.6920
Iteration 5000: Loss = 0.6872
Iteration 6000: Loss = 0.6540
Iteration 7000: Loss = 0.5796
Iteration 8000: Loss = 0.5065
Iteration 9000: Loss = 0.4463
Iteration 10000: Loss = 0.3974

Training Completed !!!
```
Training loss over epochs:
<img width="855" height="547" alt="image" src="https://github.com/user-attachments/assets/c152d791-78ee-43ad-957f-72a3e9e6cc7e" />

XOR gate predictions:
```ipynb
========================================
XOR Gate Predictions:
========================================
Input  Output  Predicted  Probability
----------------------------------------
[1, 0]    1      1        0.6943
[1, 1]    0      0        0.4253
[0, 1]    1      1        0.6945
[0, 0]    0      0        0.2638
========================================
```

Final weights:
```ipynb
Final Weights (Input to Hidden):
 [[0.6945972  0.69298621]
 [4.95603514 4.90910154]]
Final Weights (Hidden to Output):
 [[-8.76869413  6.71602913]]
```

<H3> Result:</H3>
Thus, XOR classification problem can be solved using MLP in Python 
