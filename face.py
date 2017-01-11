from PIL import Image
import numpy as np
from numpy import linalg as LA
import random

path = "./orl_faces/"

ratio = 1#3.5
w, h = int(round(92 / ratio)),int(round(112 / ratio))
def GetData(filename):
    #im = Image.open("./orl_faces/s1/1.pgm")
    im = Image.open(filename)
    im = im.resize((w, h))
    data = im.getdata()
    a = np.array(data)
    return a

n = 40 
people = [[] for _ in range(n)]
for i in range(n):
    for j in range(10):
        name = path + "s%d/%d.pgm" % (i + 1, j + 1)
        people[i].append(GetData(name))

# Train
X = np.matrix(np.zeros((n * 7, w * h)))
k = 0
S = np.matrix(np.zeros((n, 10)))
ps = np.load("s.npy")
for i in range(n):
    #s = [e for e in range(10)]
    #random.shuffle(s)
    s = ps[i, :]
    #S[i, :] = s 
    for j in s[:7]:
        X[k, :] = people[i][int(j)]
        k += 1

def Train():
    #avg 
    avg = np.mean(X)
    W = X - avg
    e,r = W.shape
    if r <= e:
        Q = W.T * W # r * r
        pw, pv = LA.eig(Q)
        #pv: (r*r)
    else:
        Q = W * W.T # e * e
        pw, ov = LA.eig(Q)
        pv = W.T * ov # (r,e) * (e * r)
    np.save("w.npy", pw)
    np.save("v.npy", pv)
    np.save("s.npy", S)

def Recognize():
    pw = np.load("w.npy")
    pv = np.load("v.npy")
    ps = np.load("s.npy")
    #pv = np.eye(w * h)
    ev = pv[:, :50]
    # X n * (wh)
    # ev wh * 80 
    Y = X * ev # 
    right = 0
    for i in range(n):
        for j in ps[i, 7:]:
            p = np.matrix(people[i][int(j)]).reshape((w * h, 1))
            y = ev.T * p
            #recognize
            f = np.tile(y.T, (n * 7, 1))
            d = Y - f
            g = np.multiply(d, d)
            su = np.sum(g, axis = 1)
            v = np.argmin(su, axis = 0)
            r = v / 7
            if r == i:
                right += 1

    acc = right * 1.0 / (n * 3)
    print ("accuracy: %f %%" % (acc * 100.0))

#Train()
Recognize()
