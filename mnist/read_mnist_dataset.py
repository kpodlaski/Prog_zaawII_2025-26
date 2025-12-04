import random
import struct
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from joblib import dump, load


def read_mnist_labels(file_path):
    labels = None
    #open file
    file = open(file_path,"rb")
    #magic number
    file.read(4)
    size = struct.unpack(">i",file.read(4))[0]
    labels = []
    for i in range(size):
        label = struct.unpack(">B",file.read(1))
        labels.append(label)
    file.close()
    return np.array(labels)


def read_mnist_images(file_path):
    images = None
    file = open(file_path, "rb")
    # magic number
    file.read(4)
    size, rows, cols = struct.unpack(">iii", file.read(12))
    #print (size, rows, cols)
    images = []
    for i in range(size):
        image = []
        for x in range(rows*cols):
            pixel = struct.unpack(">B", file.read(1))
            image.append(pixel)
        images.append(np.array(image, dtype='float'))
    file.close()
    return np.array(images), rows, cols

def show_sample(labels, images):
    r = 3
    c = 5
    fig, axis = plt.subplots(r, c)
    for x in range(r):
        for y in range(c):
            id = random.randrange(len(labels))
            axis[x, y].imshow(images[id].reshape((rows, cols)), cmap='gray')
            axis[x, y].set_title(str(labels[id]))
    print("Image is ready")
    plt.savefig("../out/img.png")
    print("Image is saved")
    # plt.show()

def model_assesment(model, patterns, labels, out_folder, name_prefix):
    print("Starting ", name_prefix, "assessment")
    predicted = model.predict(patterns)
    print("Finished prediction ", name_prefix)
    np.savetxt(out_folder+name_prefix+"_predicted.csv", predicted.astype(int), fmt='%i', delimiter=";")
    cm = confusion_matrix(y_true=labels, y_pred=predicted)
    np.savetxt(out_folder+name_prefix+"_confusion_matrix.csv", cm.astype(int), fmt='%i', delimiter=";")
    print("Obtained acc for ", name_prefix, np.trace(cm) / np.sum(cm))
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig(out_folder+name_prefix+"_confusion_matrix.png")


test_labels = read_mnist_labels("/_data/mnist/t10k-labels-idx1-ubyte")
train_labels = read_mnist_labels("/_data/mnist/train-labels-idx1-ubyte")
# print(len(test_labels))
# print(len(train_labels))
test_images, rows, cols = read_mnist_images("/_data/mnist/t10k-images-idx3-ubyte")
train_images, rows, cols = read_mnist_images("/_data/mnist/train-images-idx3-ubyte")
#print(test_images.shape)
#show_sample(test_labels, test_images)



model = svm.SVC()
no_samples = train_images.shape[0]
train_data = train_images.reshape((no_samples,-1))
no_samples = test_images.shape[0]
test_data = test_images.reshape((no_samples,-1))
if not os.path.exists('../out/model_svm.joblib'):
    print("Training a new model")
    model.fit(train_data, train_labels.reshape(-1))
    # zapis modelu do pliku
    dump(model, '../out/model_svm.joblib')
else:
    print("Reading model from file")
    model = load('../out/model_svm.joblib')
    print(model)

model_assesment(model, test_data,test_labels, "../out/svm/", "test")
model_assesment(model, train_data,train_labels, "../out/svm/", "train")


