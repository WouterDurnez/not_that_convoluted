import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.animation as ani

if __name__ == '__main__':

    # Create some dummy data: we establish a linear relationship between x and y
    a = np.random.rand()
    b = np.random.rand()

    a=0

    x = np.linspace(start=0, stop=100, num=100)
    y = a * x + b

    # Now let's create some noisy measurements
    noise = np.random.normal(size=100)
    y_noisy = a * x + b + noise

    # What's the overall error?
    mse_actual = np.sum(np.power(y-y_noisy,2))/len(y)

    # Visualize
    plt.scatter(x,y_noisy, label='Measurements', alpha=.7)
    plt.plot(x,y,'r', label='Underlying')
    plt.legend()
    plt.show()

    # Let's learn something!
    inputs = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(1)
    targets = torch.from_numpy(y_noisy).type(torch.FloatTensor).unsqueeze(1)


    # This is our model (one hidden node + bias)
    model = torch.nn.Linear(1,1)
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
    loss_function = torch.nn.MSELoss()

    # What does it predict right now?
    shuffled_inputs, preds = [], []
    for input, target in zip(inputs,targets):

        pred = model(input)
        shuffled_inputs.append(input.detach().numpy()[0])
        preds.append(pred.detach().numpy()[0])

    # Visualize
    plt.scatter(x,y_noisy, color='blue', label='Measurements', alpha=.7)
    plt.plot(shuffled_inputs, preds, color='orange', label='Predictions', alpha=.7)
    plt.plot(x,y,'r', label='Underlying')
    plt.legend()
    plt.show()

    # Let's train!
    epochs = 100
    a_s, b_s = [], []

    for epoch in range(epochs):

        # Reset optimizer values
        optimizer.zero_grad()

        # Predict values using current model
        preds = model(inputs)

        # How far off are we?
        loss = loss_function(targets,preds)

        # Calculate the gradient
        loss.backward()

        # Update model
        optimizer.step()

        for p in model.parameters():
            print('Grads:', p.grad)

        # New parameters
        a_s.append(list(model.parameters())[0].item())
        b_s.append(list(model.parameters())[1].item())

        print(f"Epoch {epoch+1} -- loss = {loss}")

    # What just happened?
    fig = plt.figure()
    ax = plt.axes(xlim=(0,100))
    ax.scatter(x, y_noisy, color='blue', label='Measurements', alpha=.7)
    ax.plot(x, y, color='red', label='Underlying')
    line, = ax.plot([],[], color='orange', label='Predictions')
    plt.legend()

    def init():
        line.set_data([],[])
        return line,

    def plot_epoch(i):

        y_temp = x * a_s[i] + b_s[i]
        line.set_data(x,y_temp)
        return line

    animation = ani.FuncAnimation(fig, plot_epoch, init_func=init, frames=100, interval=100)
    animation.save(os.path.join('visuals','linear_regression_SGD.gif'), writer='ffmpeg')
    plt.show()
