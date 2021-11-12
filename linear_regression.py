#    _  _     _     _   _         _                        _      _          _
#   | \| |___| |_  | |_| |_  __ _| |_   __ ___ _ ___ _____| |_  _| |_ ___ __| |
#   | .` / _ \  _| |  _| ' \/ _` |  _| / _/ _ \ ' \ V / _ \ | || |  _/ -_) _` |
#   |_|\_\___/\__|  \__|_||_\__,_|\__| \__\___/_||_\_/\___/_|\_,_|\__\___\__,_|
#

"""
Linear regression toy example
"""

import os

import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.linear_model import LinearRegression
from helper import blue,cinnabar,celeste,mint

if __name__ == '__main__':

    # Create some dummy data: we establish a linear relationship between x and y
    n = 500
    noise = .1
    a = 2*(np.random.rand()-.5)
    b = np.random.rand()

    x = np.linspace(start=0, stop=1, num=n)
    y = a * x + b

    # Now let's create some noisy measurements
    e = noise * np.random.normal(size=n)
    y_noisy = a * x + b + e

    # What's the overall error?
    mse_actual = np.sum(np.power(y - y_noisy, 2)) / len(y)

    # Visualize
    plt.scatter(x, y_noisy, color=celeste, label='Measurements')
    plt.plot(x, y, color=blue, lw=3, label='Underlying')
    plt.legend()
    plt.show()

    # Let's learn something!
    inputs = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(1)
    targets = torch.from_numpy(y_noisy).type(torch.FloatTensor).unsqueeze(1)

    # This is our model (one hidden node + bias)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    loss_function = torch.nn.MSELoss()

    # What does it predict right now?
    shuffled_inputs, preds = [], []
    for input, target in zip(inputs, targets):
        pred = model(input)
        shuffled_inputs.append(input.detach().numpy()[0])
        preds.append(pred.detach().numpy()[0])

    # Visualize
    sns.despine(left=True, bottom=True)
    plt.scatter(x, y_noisy, edgecolors='none', color=celeste, label='Measurements', alpha=.7)
    plt.plot(shuffled_inputs, preds, color=blue, label='Predictions', alpha=.7)
    plt.plot(x, y, color=cinnabar, label='Underlying')
    plt.legend()
    plt.show()

    # Let's train!
    epochs = 1000
    a_s, b_s = [], []

    # Loop over epochs (batch size = all data)
    for epoch in range(epochs):

        # Reset optimizer values
        optimizer.zero_grad()

        # Predict values using current model
        preds = model(inputs)

        # How far off are we?
        loss = loss_function(targets, preds)

        # Calculate the gradient
        loss.backward()

        # Update model
        optimizer.step()

        # New parameters
        a_s.append(list(model.parameters())[0].item())
        b_s.append(list(model.parameters())[1].item())

        print(f"Epoch {epoch + 1} -- loss = {loss}")

    # What just happened?
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1))
    sns.despine(left=True, bottom=True)
    ax.scatter(x, y_noisy, color=celeste, label='Measurements', alpha=.4)
    ax.plot(x, y, color=blue, linewidth=3, label='Underlying')
    line, = ax.plot([], [], color=cinnabar, linewidth=3, label='Predictions')
    plt.legend()

    def init():
        line.set_data([], [])
        return line,

    def plot_epoch(i):

        y_temp = x * a_s[i] + b_s[i]
        line.set_data(x, y_temp)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [labels[0], f'{labels[1]} - epoch {i + 1}', labels[2]])
        return line

    animation = ani.FuncAnimation(fig, plot_epoch, init_func=init, frames=500, interval=1)
    animation.save(os.path.join('visuals', 'linear_regression_SGD.gif'), writer='ffmpeg')
    plt.show()

    # How does this compare to the traditional approach?
    lr = LinearRegression()
    lr.fit(x.reshape(-1, 1), y_noisy.reshape(-1, 1))
    a_alt = lr.coef_
    b_alt = lr.intercept_
