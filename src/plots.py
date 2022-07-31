import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def plot_line(x, y, label, color1, color2):
    #plt.plot(x, y, color=color1, label=label)
    y = savgol_filter(y, 100, 3)
    plt.plot(x, y, color=color2, label=label + " Avg.")

def plot_loss(session_dir, loss_gen, loss_critic):
    plt.figure()
    epochs = len(loss_gen)
    epoch_range = np.arange(1, epochs + 1)
    plot_line(epoch_range, loss_gen, "Generator Loss", "navy", "cornflowerblue",)
    plot_line(epoch_range, loss_critic, "Critic Loss", "maroon", "lightcoral")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid()
    plt.savefig(session_dir + '/plots/loss.png')
    plt.close()

def plot_fid(session_dir, fid_scores):
    plt.figure()
    epochs = len(fid_scores)
    epoch_range = np.arange(1, epochs + 1)
    #plot_line(epoch_range, fid_scores, "FID Score", "navy", "cornflowerblue")
    plt.plot(epoch_range, fid_scores, color="cornflowerblue", label="FID Score")
    plt.ylim([0, 50])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid()
    plt.savefig(session_dir + '/plots/fid.png')
    plt.close()
    