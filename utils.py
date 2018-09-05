from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
import numpy as np


def visualize_2d(embedding, label, domain, class_num):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(0.0, 1.0, class_num))

    xx = embedding[:, 0]
    yy = embedding[:, 1]

    for i in range(class_num):
        ax.scatter(xx[label == i], yy[label == i],
                   color=colors[i], s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig("TSNE_Label_2D.pdf", format='pdf', dpi=600)
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(0.0, 1.0, class_num))

    for i in range(2):
        ax.scatter(xx[domain == i], yy[domain == i],
                   color=cm.bwr(i/1.), s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig("TSNE_Domain_2D.pdf", format='pdf', dpi=600)
    plt.show()
    plt.close()


def visualize_3d(embedding, label, domain, class_num):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = cm.rainbow(np.linspace(0.0, 1.0, class_num))

    xx = embedding[:, 0]
    yy = embedding[:, 1]
    zz = embedding[:, 2]

    for i in range(class_num):
        ax.scatter(xx[label == i], yy[label == i],
                   zz[label == i], color=colors[i], s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.zaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig("TSNE_Label_3D.pdf", format='pdf', dpi=600)
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in range(2):
        ax.scatter(xx[domain == i], yy[domain == i],
                   zz[domain == i], color=cm.bwr(i/1.), s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.zaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig("TSNE_Domain_3D.pdf", format='pdf', dpi=600)
    plt.show()
    plt.close()
