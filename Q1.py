import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

(x,y),(a,b) = fashion_mnist.load_data()
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
for i in range(10):
    plt.subplot(2, 5, i + 1)
    img = x[y==i][0]
    plt.imshow(img,cmap='gray')
    plt.title(labels[i])
    plt.axis('off')
plt.tight_layout()
plt.show()