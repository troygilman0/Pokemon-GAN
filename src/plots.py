import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_disc, loss_gen):
    epochs = len(loss_disc)
    epoch_range = np.arange(1, epochs + 1)
    plt.plot(epoch_range, loss_disc, color='r', label="Discriminator Loss")
    plt.plot(epoch_range, loss_gen, color='b', label="Generator Loss")
    plt.legend()
    plt.grid()
    plt.savefig('plots/loss.png')
    plt.show()
    