import matplotlib.pyplot as plt


def plot_accuracy(train_accuracy, test_accuracy, path,baseline=None,xmax=20):
    # Accuracy
    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    if baseline is not None:
        plt.hlines(baseline, 0, xmax, 'g')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    if baseline is not None:
        plt.legend(['train', 'validation', 'baseline'], loc='lower right')
    else:
        plt.legend(['train', 'validation'], loc="lower right")
    plt.savefig(path+'accuracy.png')
    plt.close()

    
def plot_loss(train_loss, test_loss, path,baseline=None,xmax=20):
    # Loss
    plt.plot(train_loss)
    plt.plot(test_loss)
    if baseline is not None:
        plt.hlines(baseline, 0, xmax, 'g')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if baseline is not None:
        plt.legend(['train', 'validation', 'baseline'], loc='upper right')
    else:
        plt.legend(['train', 'validation'], loc="upper right")
    plt.savefig(path+'loss.png')
    plt.close()   
    
