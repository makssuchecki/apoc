import numpy as np
from PIL import Image

S = 10
img = np.array(Image.open("./potworek_pixelart.png").convert("RGB"), dtype=np.float64)
h, w, _ = img.shape
H, W = h * S, w * S

# współrzędne w obrazie wejściowym (środki pikseli, half-pixel)
X, Y = np.meshgrid(np.arange(W), np.arange(H))
u = (X + 0.5) / S - 0.5
v = (Y + 0.5) / S - 0.5

def clamp_x(a): return np.clip(a, 0, w - 1)
def clamp_y(a): return np.clip(a, 0, h - 1)

def save(arr, name):
    Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8)).save(name)

# (a) Nearest neighbor
xn = clamp_x(np.floor(u + 0.5)).astype(int)
yn = clamp_y(np.floor(v + 0.5)).astype(int)
save(img[yn, xn], "a_nearest.png")

# (b) średnia dwóch najbliższych 
iu = np.floor(u + 0.5).astype(int)      # round half up
iv = np.floor(v + 0.5).astype(int)
dx, dy = u - iu, v - iv
ju, jv = iu.copy(), iv.copy()
horiz = np.abs(dx) > np.abs(dy)
ju = np.where(horiz, iu + np.where(dx > 0, 1, -1), iu)
jv = np.where(~horiz, iv + np.where(dy > 0, 1, -1), iv)
iu, ju = clamp_x(iu).astype(int), clamp_x(ju).astype(int)
iv, jv = clamp_y(iv).astype(int), clamp_y(jv).astype(int)
save((img[iv, iu] + img[jv, ju]) / 2, "b_srednia2.png")

# 4 sąsiadów (wspólne dla c i d)
x0 = np.floor(u).astype(int)
y0 = np.floor(v).astype(int)
a = (u - x0)[..., None]
b = (v - y0)[..., None]
xa, xb = clamp_x(x0).astype(int), clamp_x(x0 + 1).astype(int)
ya, yb = clamp_y(y0).astype(int), clamp_y(y0 + 1).astype(int)
p00, p01 = img[ya, xa], img[ya, xb]
p10, p11 = img[yb, xa], img[yb, xb]

# (c) bilinear
bil = (1-a)*(1-b)*p00 + a*(1-b)*p01 + (1-a)*b*p10 + a*b*p11
save(bil, "c_bilinear.png")

# (d) (max+min)/2 z 4 sąsiadów, osobno dla R, G, B
stack = np.stack([p00, p01, p10, p11])           # (4, H, W, 3)
save((stack.max(axis=0) + stack.min(axis=0)) / 2, "d_maxmin.png")