
import matplotlib.pyplot as plt


def perform_loss_cof(y1 , y2 ):
    """Define view.
        Arguments:
            y1 -- loss
            y2 -- cof
    """
    plt.subplot(1, 2, 1)
    x1 = range(0, len(y1))
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.subplot(2, 1, 2)
    plt.plot(x1, y2, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.savefig('plot.svg', format='svg')
