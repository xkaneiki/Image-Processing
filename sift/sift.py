import numpy as np
import cv2
import math


def togray(src):
    return src[:, :, 0]*0.299+src[:, :, 1]*0.587+src[:, :, 2]*0.144


def get_sigma(sigma0, o, s, S):
    return sigma0*math.pow(2, o+s/S)


def gaussfilter(src, sigma):
    r = (int)(sigma*3)*2+1

    return cv2.GaussianBlur(src, (r, r), sigma)


def resize(src, shape):
    return cv2.resize(src, (shape[1], shape[0]))


def octor_gaussain(src, sigma0, o, S):
    res = src[:, :, np.newaxis].repeat(S+3, axis=2)
    for s in range(S+3):
        sigma = get_sigma(sigma0, o, s, S)
        # print(sigma)
        res[:, :, s] = gaussfilter(res[:, :, s], sigma)
    return res


def Gaussian(src, sigma0, O, S):
    raw = resize(src, (src.shape[0]*2, src.shape[1]*2))
    gauss = []
    for o in range(O):
        gauss.append(octor_gaussain(raw, sigma0, o, S))
        raw = gauss[-1][:, :, -3]
        raw = resize(raw, (raw.shape[0]//2, raw.shape[1]//2))
    return gauss


def octor_dog(g_o, S):
    return g_o[:, :, 1:S+3]-g_o[:, :, 0:S+2]


def DOG(gauss, O, S):
    dog = []
    for o in range(O):
        dog.append(octor_dog(gauss[o], S))
    return dog


# 极值检测
def exact(d, x, y, s):

    dxx = d[x+1, y, s]+d[x-1, y, s]-2*d[x, y, s]
    dyy = d[x, y+1, s]+d[x, y-1, s]-2*d[x, y, s]
    dss = d[x, y, s+1]+d[x, y, s-1]-2*d[x, y, s]
    dxy = d[x+1, y+1, s]+d[x-1, y-1, s]-d[x+1, y-1, s]-d[x-1, y+1, s]
    dxs = d[x+1, y, s+1]+d[x-1, y, s-1]-d[x+1, y, s-1]-d[x-1, y, s+1]
    dys = d[x, y+1, s+1]+d[x, y-1, s-1]-d[x, y+1, s-1]-d[x, y-1, s+1]
    dx = (d[x+1, y, s]-d[x-1, y, s])/2
    dy = (d[x, y+1, s]-d[x, y-1, s])/2
    ds = (d[x, y, s+1]-d[x, y, s-1])/2

    hessian2 = np.array([
        [dxx, dxy],
        [dxy, dyy],
    ], np.float)

    if np.linalg.det(hessian2) == 0 or (dxx+dyy)*(dxx+dyy)/np.linalg.det(hessian2) >= 11*11/10:
        return False, []

    DX2 = hessian3 = np.array([
        [dxx, dxy, dxs],
        [dxy, dyy, dys],
        [dxs, dys, dss]
    ], np.float)

    DX = np.array([
        [dx, dy, ds]
    ]).T

    dX = -np.matmul(np.linalg.inv(DX2), DX)

    if dX[0, 0] > 0.5:
        x += 1
    elif dX[0, 0] < -0.5:
        x -= 1

    if dX[1, 0] > 0.5:
        y += 1
    elif dX[1, 0] < -0.5:
        y -= 1

    if dX[2, 0] > 0.5:
        s += 1
    elif dX[2, 0] < -0.5:
        s -= 1

    if x < 1 or x >= d.shape[0]-1 or y < 1 or y >= d.shape[1]-1 or s < 1 or s >= d.shape[2]-1 or d[x, y, s] < 0.03:
        return False, []

    return True, [x, y, s]


def get_extreme_0(dog, O, S):
    extreme0 = []
    for o in range(O):
        for s in range(1, S+1):
            for x in range(1, dog[o].shape[0]-1):
                for y in range(1, dog[o].shape[1]-1):
                    if (dog[o][x, y, s] >= np.max(dog[o][x-1:x+2, y-1:y+2, s-1]) and dog[o][x, y, s] >= np.max(dog[o][x-1:x+2, y-1:y+2, s]) and dog[o][x, y, s] >= np.max(
                            dog[o][x-1:x+2, y-1:y+2, s+1])
                        ) or (
                       dog[o][x, y, s] <= np.min(dog[o][x-1:x+2, y-1:y+2, s-1]) and dog[o][x, y, s] <= np.min(dog[o][x-1:x+2, y-1:y+2, s]) and dog[o][x, y, s] <= np.min(
                            dog[o][x-1:x+2, y-1:y+2, s+1])):

                        # if is_exact(dog[o], x, y, s):
                        if(dog[o][x, y, s] >= 0.03):
                            extreme0.append([x, y, s, o])

    return extreme0

# 删除不好的特征点


def get_extreme_1(dog, O, S, extreme0):
    extreme = []
    for i in extreme0:
        o = i[3]
        s, e = exact(dog[i[3]], i[0], i[1], i[2])
        if s:
            e.append(o)
            extreme.append(e)
    return extreme


def get_M_D(dog, sigma0, O, S, extreme):
    res = []
    for e in extreme:
        x, y, s, o = e
        sigma = get_sigma(sigma0, o, s, S)
        r = (int)(3*(sigma))
        d = dog[o][:, :, s]
        h, w = d.shape
        dx = np.zeros(d.shape, np.float)
        dx[1:h-1, :] = d[2:h, :]-d[0:h-2, :]
        dy = np.zeros(d.shape, np.float)
        dy[:, 1:w-1] = d[:, 2:w]-d[:, 0:w-2]
        m = np.sqrt(dx*dx+dy*dy)
        pi = math.acos(-1)
        cm = np.zeros((36), np.float)

        for i in range(max(0, x-2*r), min(h, x+2*r+1)):
            for j in range(max(0, y-2*r), min(w, y+2*r+1)):
                if dx[i, j] == 0:
                    ag = math.atan(dy[i, j]/(dx[i, j]+0.003))/pi*180
                else:
                    ag = math.atan(dy[i, j]/dx[i, j])/pi*180
                if dx[i, j] < 0:
                    ag += 180
                cm[(int)(ag//10)] += m[i][j]

        mx = np.max(cm)
        for i in range(36):
            if cm[i] >= 0.8*mx:
                res.append((x, y, s, o, i, cm[i]))
    return res


def get_character(dog, pot, sigma0, O, S):
    character = []
    for p in pot:
        x, y, s, o, di, l = p
        sigma = get_sigma(sigma0, o, s, S)
        r = (int)(3*sigma*math.sqrt(2)*(4+1)/2+0.5)
        d = dog[o][:, :, s]
        h, w = d.shape
        dx = np.zeros(d.shape, np.float)
        dx[1:h-1, :] = d[2:h, :]-d[0:h-2, :]
        dy = np.zeros(d.shape, np.float)
        dy[:, 1:w-1] = d[:, 2:w]-d[:, 0:w-2]
        m = np.sqrt(dx*dx+dy*dy)

        pi = math.acos(-1)
        cm = np.zeros((36), np.float)

        v = np.zeros((4, 4, 8), np.float)
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                if x+i < 0 or x+i >= h or y+j < 0 or y+j >= w:
                    continue
                n = np.array([math.cos(di/18*pi)*i-math.sin(di/18*pi)*j,
                              math.sin(di/18*pi)*i+math.cos(di/18*pi)*j], np.int)

                pos = (n/(3*sigma)+4/2).astype(np.int)
                if pos[0] >= 4 or pos[0] < 0 or pos[1] >= 4 or pos[1] <= 0:
                    continue
                w = m[x+i, y+j]*math.exp(-(i*i+j*j)/(2*(0.5*4)*(0.5*4)))

                if dx[x+i, y+j] == 0:
                    ag = math.atan(dy[x+i, y+j]/(dx[x+i, y+j]+0.003))/pi*180
                else:
                    ag = math.atan(dy[x+i, y+j]/dx[x+i, y+j])/pi*180
                if dx[x+i, y+j] < 0:
                    ag += 180

                dr = ((int)((ag+di*10)//45)) % 8
                # print(pos, dr)
                v[pos[0], pos[1], dr] += w
        v = np.reshape(v, (1, 128))
        print(np.sum(v))
        v = v/np.sqrt(np.sum(v)+0.2)
        character.append(v)

    return np.vstack(character)


def sift(src, sigma0, S):
    h, w = src.shape
    O = min(math.floor(math.log2(h)), math.floor(math.log2(w)))-3
    O = max(1, (int)(O))
    gauss = Gaussian(src, sigma0, O, S)
    # for g in gauss:
    #     print(g.shape)

    dog = DOG(gauss, O, S)
    # for d in dog:
    #     print(d.shape)
    #     print(d)

    # show = []
    # for i in range(5):
    #     show.append(dog[1][:, :, i])
    # show = np.hstack(show)
    # cv2.imshow("show", show.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyWindow()

    # (x,y,sigma)=(x,y,o,s)
    extreme = get_extreme_0(dog, O, S)
    # for e in extreme0:
    #     print(e)
    print(len(extreme))
    for i in range(5):
        extreme = get_extreme_1(dog, O, S, extreme)

    print(len(extreme))

    pot = get_M_D(dog, sigma0, O, S, extreme)

    print(len(pot))

    character = get_character(dog, pot, sigma0, O, S)

    return extreme, pot, character


if __name__ == '__main__':
    # print(math.cos(90))
    # print(math.floor(math.log2(3)))
    # print((int)(0.3))
    img = cv2.imread("imgs/jobs.jpg")
    img_ = togray(np.array(img, np.float))

    extreme, pot, character = sift(img_, 1.6, 3)
    print(character, character.shape)
    img = cv2.cvtColor(img_.astype(np.uint8), cv2.COLOR_BGR2RGB)
    img1 = img.copy()
    # img = resize(img,(img.shape[0]*2,img.shape[1]*2))
    for i in extreme:
        if(i[3] > 1):
            break
        if i[3] == 1:
            cv2.circle(img, (i[1], i[0]), 5, (255, 0, 0))

    pi = math.acos(-1)
    for p in pot:
        x, y, s, o, d, l = p
        if(o > 1):
            break
        if o == 1:
            nx = (int)(x+math.cos(d/18*pi)*l/10)
            ny = (int)(y+math.sin(d/18*pi)*l/10)
            if d >= 18:
                d1 = (d-18)*10
            else:
                d1 = (d+18)*10
            apha = 15
            t1 = ((int)(ny+math.sin((d1+apha)/180*pi)*l/70),
                  (int)(nx+math.cos((d1+apha)/180*pi)*l/70))
            t2 = ((int)(ny+math.sin((d1-apha)/180*pi)*l/70),
                  (int)(nx+math.cos((d1-apha)/180*pi)*l/70))
            cv2.line(img1, (y, x), (ny, nx), (255, 0, 0))
            cv2.line(img1, (ny, nx), t1, (255, 0, 0))
            cv2.line(img1, (ny, nx), t2, (255, 0, 0),)

    cv2.imshow("img", np.hstack([img, img1]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
