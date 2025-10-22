import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('../images/wiewiorka.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
#res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h = img2.copy()
h[:,:,0]=0
h[:,:,2]=0
# cv.imshow('Hframe', h)
# cv.waitKey(0)
plt.figure()
colors = ['green','blue','red']
for c in [0,1,2]:
    hist = np.histogram(img[:,:,c],bins=256)[0]
    plt.plot(hist, color=colors[c],  label=colors[c])
plt.legend()
plt.savefig("../out/w_hist.jpg")
plt.figure()

labels = ['H',"S", "V"]
for c in [0,1,2]:
    hist = np.histogram(img2[:,:,c],bins=256)[0]
    plt.plot(hist, color=colors[c],  label=labels[c])
plt.legend()
plt.savefig("../out/w_hist_hsv.jpg")

#plt.show()

red = img[:,:,2]
red[ (red<20)  | (red>230) ] = 0
img[:,:,2] = red
cv.imwrite("../out/w.jpg",img)
