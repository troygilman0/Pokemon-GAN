import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_disc, loss_gen, lr, critic_it):
    epochs = len(loss_disc)
    epoch_range = np.arange(1, epochs + 1)
    plt.plot(epoch_range, loss_gen, color='b', label="Generator Loss")
    plt.plot(epoch_range, loss_disc, color='r', label="Discriminator Loss")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid()
    plt.title(f'LR: {lr} Critic It: {critic_it}')
    plt.savefig('plots/loss.png')
    #plt.show()
    