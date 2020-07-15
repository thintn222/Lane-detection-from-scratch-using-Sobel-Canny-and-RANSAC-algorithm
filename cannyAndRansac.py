import imageio
import numpy as np 
import matplotlib.pyplot as plt
import random

def Zero_padding(image, half_ker_h, half_ker_w):
    zero_col = np.zeros((image.shape[0],half_ker_w))
    image = np.append(np.append(zero_col,image,axis = 1),zero_col,axis = 1)
    zero_row = np.zeros((half_ker_h, image.shape[1]))
    return np.append(np.append(zero_row,image,axis = 0),zero_row,axis = 0)

def Convol(image, kernel):
    half_ker_h = (kernel.shape[0]-1) // 2
    half_ker_w = (kernel.shape[1]-1) // 2
    # output = np.zeros((image.shape[0] - 2*half_ker_h, image.shape[1] - 2*half_ker_w))
    output = np.zeros(image.shape)
    image = Zero_padding(image, half_ker_h, half_ker_w)
    for m in range(half_ker_h, image.shape[0]-half_ker_h):
        for n in range(half_ker_w, image.shape[1]-half_ker_w):
            for i in (range(-half_ker_h, half_ker_h+1)):
                for j in (range(-half_ker_w, half_ker_w+1)):
                    output[m-half_ker_h][n-half_ker_w] += image[m+i][n+j] * kernel[i+half_ker_h][j+half_ker_w]
    return output

def Sobel(image):
    Sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Gx = Convol(image, Sobel_x)
    Gy = Convol(image, Sobel_y)
    # return np.abs(Gx) + np.abs(Gy), np.arctan2(Gy, Gx)
    return Normalize(np.sqrt(np.square(Gx) + np.square(Gy))), np.arctan2(Gx,Gy)

def Gaussian(size = 3,sigma = 1.4):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    return  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

def Non_max_suppression(image, angle):
    output = np.zeros(image.shape)
    angle[angle < 0] += np.pi
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            q = 255
            r = 255
            #angle 0
            if (0 <= angle[i,j] < np.pi/8) or (7*np.pi/8 <= angle[i,j] <= np.pi):
                q = image[i+1, j]
                r = image[i-1, j]
            #angle 45
            elif (np.pi/8 <= angle[i,j] < 3*np.pi/8):
                q = image[i+1, j+1]
                r = image[i-1, j-1]
            #angle 90
            elif (3*np.pi/8 <= angle[i,j] < 5*np.pi/8):
                q = image[i, j+1]
                r = image[i, j-1]
            #angle 135
            elif (5*np.pi/8 <= angle[i,j] < 7*np.pi/8):
                q = image[i-1, j+1]
                r = image[i+1, j-1]
            if (image[i,j] >= q) and (image[i,j] >= r):
                output[i,j] = image[i,j]
    return output

def Normalize(image):
    # return ((image - np.min(image))/(np.max(image) - np.min(image)))*255
    return (image / image.max()) * 255

def Thresholding(image, lowThresholdRatio=0.05, highThresholdRatio=0.2):
    highThreshold = 255 * highThresholdRatio
    lowThreshold = 255 * lowThresholdRatio;
    output = np.zeros(image.shape, dtype=np.int32)
    strong_i, strong_j = np.where(image >= highThreshold)
    # zeros_i, zeros_j = np.where(image < lowThreshold)
    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))
    output[strong_i, strong_j] = np.int32(255)
    output[weak_i, weak_j] = np.int32(25)
    return output

def Canny(image):
    image = Convol(image, Gaussian())
    image, theta = Sobel(image)
    image = Non_max_suppression(image, theta)
    return Thresholding(image)

def Straight1(xa, ya, xb, yb, threshold = 0):
    return np.abs(-yb*xa+xb*ya) <= threshold; # Check if vector a,b is nearly straigth.

def Straight2(x, y, a, b, c, threshold = 0):
    return np.abs(a * x + b * y + c) <= threshold



def Ransac1(image):
    max_iter = 500
    min_distance = 30
    min_i = 50
    white_i, white_j = np.where(image >= 205)
    # check = np.zeros(np.shape(white_i))
    iter = 0
    max_count = 0
    while iter < max_iter:
        iter += 1
        randid = random.sample(range(np.shape(white_i)[0]), 2)
        id1 = randid[0]
        id2 = randid[1]
        xa = white_i[id2] - white_i[id1]
        ya = white_j[id2] - white_j[id1]
        # if (abs(xa) > min_distance)and(abs(ya) > min_distance):
        if (abs(xa) > min_distance)and(abs(ya) > min_distance)and(white_i[id1]>min_i)and(white_i[id2]>min_i):
            count = 0
            for id3 in range(np.shape(white_i)[0]):
                xb = white_i[id3] - white_i[id1]
                yb = white_j[id3] - white_j[id1]
                if Straight1(xa,ya,xb,yb):
                    count += 1
            if (max_count < count):
                max_count = count
                outid1 = id1
                outid2 = id2
    return white_i[outid1], white_j[outid1], white_i[outid2], white_j[outid2]

def Ransac2(image):
    max_iter = 500
    min_distance = 30
    min_i = 50
    white_i, white_j = np.where(image >= 205)
    # check = np.zeros(np.shape(white_i))
    iter = 0
    max_count = 0
    while iter < max_iter:
        iter += 1
        randid = random.sample(range(np.shape(white_i)[0]), 2)
        id1 = randid[0]
        id2 = randid[1]
        
        # if (abs(xa) > min_distance)and(abs(ya) > min_distance):
        if (abs(white_i[id2] - white_i[id1]) > min_distance)and(abs(white_j[id2] - white_j[id1]) > min_distance)and(white_i[id1]>min_i)and(white_i[id2]>min_i):
            a, b, c = line_param(white_i[id1], white_j[id1], white_i[id2], white_j[id2])
            count = 0
            for id3 in range(np.shape(white_i)[0]):
                x = white_i[id3] - white_i[id1]
                y = white_j[id3] - white_j[id1]
                if Straight2(x, y, a, b, c):
                    count += 1
            if (max_count < count):
                max_count = count
                outid1 = id1
                outid2 = id2
    return white_i[outid1], white_j[outid1], white_i[outid2], white_j[outid2]

# def bang(image, a, b):
#     N = 5
#     for i in range(-N,N+1):
#         for j in range(-N,N+1):
#             if (a+i>0)and(a+i<254)and(b+j>0)and(b+j<668):
#                 image[a+i][b+j] = 255
#     return image

# def line(x,xa,ya,xb,yb):
#     if xa == xb:
#         y = np.zeros(np.shape(x))
#         for i in range(np.shape(x)[0]):
#             y[i] = yb
#         return y
#     return ya + (x - xa) * (yb - ya) / (xb - xa)

def line_param(xa, ya, xb, yb):
    return yb - ya, xa - xb, xa*(ya - yb) + ya*(xb - xa);

def Delete_line(image, a, b, c, threshold = 10):
    white_i, white_j = np.where(image >= 200)
    # a = yb - ya
    # b = xa - xb
    # c = xa * (ya - yb) + ya * (xb - xa)
    for i in range(np.shape(white_i)[0]):
        if Straight2(white_i[i], white_j[i], a, b, c, threshold=threshold):
            image[white_i[i]][white_j[i]] = 0
    return image
        

def Draw(image, a, b, c):
    y = np.linspace(0,np.shape(image)[1])
    f = lambda y: - a * y / b - c / b
    x = f(y)
    axe = plt.gca()
    axe.set_xlim([0,np.shape(image)[1]])
    axe.set_ylim([0,np.shape(image)[0]])
    axe.invert_yaxis()
    plt.plot(x,y,'-',color='r')
    plt.imshow(image, cmap='gray')
    plt.show()

def Draw_lines(image, n_line = 3):
    temp_image = image
    params = []
    for i in range(n_line):
        xa, ya, xb, yb = Ransac1(temp_image)
        a, b, c = line_param(xa, ya, xb, yb)
        params.append([a, b, c])
        temp_image = Delete_line(temp_image, a, b, c, threshold=40)
    y = np.linspace(0, np.shape(image)[1])
    plt.imshow(image, cmap='gray')
    axe = plt.gca()
    for param in params:
        f = lambda y: -param[0]*y/param[1] - param[2]/param[1]
        x = f(y)
        axe.set_xlim([0,np.shape(image)[1]])
        axe.set_ylim([0,np.shape(image)[0]])
        axe.invert_yaxis()
        plt.plot(x,y,'-',color='r')
    plt.show()



def main():
    print("Main...")
    image = np.array(imageio.imread("lane.png")[:,:,0])

    image = Canny(image)

    Draw_lines(image, n_line=4)

    # xa, ya, xb, yb = Ransac2(image)
    # a, b, c = line_param(xa, ya, xb, yb)
    # image_red = bang(image,xa,ya)
    # image_red = bang(image_red,xb,yb)
    # image_red = np.array(image)

    # image_red = Delete_line(image_red, a, b, c, threshold = 50)
    # Draw(image_red, a, b, c)

    # xa, ya, xb, yb = Ransac1(image_red)
    # a, b, c = line_param(xa, ya, xb, yb)
    # image_red = Delete_line(image_red, a, b, c, threshold = 50)
    # Draw(image_red, a, b, c)

if __name__ == '__main__':
    main()