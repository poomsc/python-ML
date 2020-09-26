from scipy.io import loadmat
import matplotlib.pyplot as plt

mnist_raw = loadmat('./mnist-original.mat')

mnist = {
    'data': mnist_raw['data'].T,
    'target': mnist_raw['label'][0]
}

x, y = mnist['data'], mnist['target']

n = 0
number = x[n]
number_image = number.reshape(28, 28)

print(x.shape)

plt.title(y[n])

plt.imshow(
    number_image,
    plt.cm.binary,
    interpolation='nearest',
    
)

plt.show()
