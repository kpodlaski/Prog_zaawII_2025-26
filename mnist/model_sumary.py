import numpy as np

def results_assesment(name, confusion_matrix_file):
    cm = np.loadtxt(confusion_matrix_file, delimiter=";")
    print("Obtained acc for ", name, "{:.4f}".format(np.trace(cm) / np.sum(cm)))

results_assesment("Train set", "../out/svm/train_confusion_matrix.csv")
results_assesment("Test set", "../out/svm/test_confusion_matrix.csv")