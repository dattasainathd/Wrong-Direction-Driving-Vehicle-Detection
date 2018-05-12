import sys
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
import time
import operator
import random
import shutil, os
from scipy import signal

######## Setting most important parameters #######

show_images_flag = False
if(len(sys.argv)<2):
    sys.exit("please provide the name of video to be analyzed")


video = sys.argv[1]
if(len(sys.argv)>2):
    show_images_flag = sys.argv[2]



####### To create an empty sub_final folder everytime we run this script #######

if not os.path.exists('sub_final'):
    os.mkdir('sub_final')

shutil.rmtree('sub_final')
os.makedirs('sub_final')


####### This code can be used to generate gray frames from given video #######

print("extracting frames from image")

cap = cv2.VideoCapture(video)

count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if(ret):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

#     cv2.imshow('frame',gray)
    cv2.imwrite("sub_final/frame%d.jpg" % count, gray)
    count = count + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

####### The above code to extract the frames from the images was given in the official documentation of opencv #######
####### Reference: https://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html #######

sub_part_of_image1 = str(int(count*0.65))
sub_part_of_image2 = str(int(count*0.65)+1)

path_of_image1 = 'sub_final/frame'+ sub_part_of_image1 + '.jpg'
path_of_image2 = 'sub_final/frame'+ sub_part_of_image2 + '.jpg'


####### Takes a matrix as input and converts range of matrix to 0 to 255 range #######
####### This function is usually used for printing purposes (better vizualization) #######

def change_range(old_array):
    new_array = np.zeros(old_array.size).reshape(old_array.shape[0],old_array.shape[1])
    max_intensity = old_array.max()
    min_intensity = old_array.min()
    for j in range(0, old_array.shape[1]):
        for i in range(0, old_array.shape[0]):
            new_array[i][j] = ((old_array[i][j]-min_intensity)*(255/(max_intensity - min_intensity)))
    return new_array

####### This function takes in an image matrix and smoothes its with k dimension kernal #######

def smooth_image(img,k):
    # creating a new image matrix in which smoothed values will be stored
    a = img.copy()
    #k = dimension of the filter
    for i in range(math.floor(k/2), (img.shape[0]-math.ceil(k/2))):
        for j in range(math.floor(k/2), (img.shape[1]-math.ceil(k/2))):
            i_lower = i-int(math.floor(k/2))
            i_upper = i+int(math.ceil(k/2))
            j_lower = j-int(math.floor(k/2))
            j_upper = j+int(math.ceil(k/2))
            sub_img_flat = img[i_lower:i_upper,j_lower:j_upper].ravel().copy()
            a[i][j] = np.sum(sub_img_flat)/(k*k)
    return a

####### This function used for reporducing edge strength and edge orientation matrix for given smoothed #######
####### matrix image.  This function returns edge strenght matrix (mapped from 0 to 255) and edge #######
####### orientation matrix mapped from -90 to 90 #######

def edge_detection(a):

    img = a
    # creating a new image matrix in which edge-y values will be stored
    c = np.zeros(img.size).reshape(img.shape[0],img.shape[1])
    c_display = np.zeros(img.size).reshape(img.shape[0],img.shape[1])

    # creating a new image matrix in which edge-x values will be stored
    d = np.zeros(img.size).reshape(img.shape[0],img.shape[1])
    d_display = np.zeros(img.size).reshape(img.shape[0],img.shape[1])

    # applying x and y edges on the image and storing the values in another matrix images
    for j in range(1, img.shape[1]-1):
        for i in range(1, img.shape[0]-1):
            c[i][j] = ((-0.5)*int(a[i-1][j]) + (0.5)*int(a[i+1][j]) + (0)*(int(a[i][j])))
            d[i][j] = ((-0.5)*int(a[i][j-1]) + (0.5)*int(a[i][j+1]) + (0)*(int(a[i][j])))

    c_display = change_range(c)

    # creating a new image matrix in whihc edge matrix values will be stored
    e = np.zeros(img.size).reshape(img.shape[0],img.shape[1])
    e_display = np.zeros(img.size).reshape(img.shape[0],img.shape[1])

    # finding edge magnitude values from edge x and edge y values
    for j in range(1, img.shape[1]-1):
        for i in range(1, img.shape[0]-1):
            e[i][j] = math.sqrt((c[i][j]*c[i][j]) + (d[i][j]*d[i][j]))

    e_display = change_range(e)

    # creating a new image matrix in whihc edge orientation matrix values will be stored
    f = np.zeros(img.size).reshape(img.shape[0],img.shape[1])
    # just trying to avoid divide by zero
    d[d == 0] = 0.0000000001

    # finding edge orientation values from edge x and edge y values
    for j in range(1, img.shape[1]-1):
        for i in range(1, img.shape[0]-1):
            f[i][j] = math.degrees(math.atan(c[i][j]/d[i][j]))
    f_display = change_range(f)
    return e_display, f

####### This is a threshold function which takes in a matrix and threshold level number #######
####### and outputs all the values greater than threshold level, retains the values in matrix #######

def threshold(e_display, level):
    image = np.zeros((e_display.shape[0], e_display.shape[1]))
    for i in range(0, e_display.shape[0]):
        for j in range(0, e_display.shape[1]):
            if(e_display[i][j]<level):
                image[i][j] = 0
            else:
                image[i][j] = e_display[i][j]
    return image

####### This is a histogram function used for display images only as it can sample for 255 bins only #######
####### Output is an array of size 255 which consists of number of pixels w.r.t. index of array #######

def histogram(img):
    bins = 255
    img_flat = img.ravel()
    max_intensity = img_flat.max()
    min_intensity = img_flat.min()
    hist = np.zeros(bins)
    for element in img_flat:
        bin_id = int(round((bins-1)*((element - min_intensity)/(max_intensity - min_intensity))))
        hist[bin_id] += 1
    return hist

####### This is a median filter with kernal size of kXk #######

def medianfilter(img,k):
    # creating a new image matrix in which smoothed values will be stored
    a = img.copy()

    #k is dimension of the filter

    for i in range(math.floor(k/2), (img.shape[0]-math.ceil(k/2))):
        for j in range(math.floor(k/2), (img.shape[1]-math.ceil(k/2))):
            i_lower = i-int(math.floor(k/2))
            i_upper = i+int(math.ceil(k/2))
            j_lower = j-int(math.floor(k/2))
            j_upper = j+int(math.ceil(k/2))
            sub_img_flat = img[i_lower:i_upper,j_lower:j_upper].ravel().copy()

            a[i][j] = np.max(sub_img_flat)
    return a

####### custom made gaussian blur kernel which inputs image, kernel size of 1-D and sigma########

def GaussianBlur_custom(image, size, sigma):
    a = image.copy()
    gaussian_kernal = signal.get_window(('gaussian',sigma),size*size)
    for i in range(math.floor(size/2), (image.shape[0]-math.ceil(size/2))):
        for j in range(math.floor(size/2), (image.shape[1]-math.ceil(size/2))):
            i_lower = i-int(math.floor(size/2))
            i_upper = i+int(math.ceil(size/2))
            j_lower = j-int(math.floor(size/2))
            j_upper = j+int(math.ceil(size/2))
            sub_img_flat = image[i_lower:i_upper,j_lower:j_upper].ravel().copy()

            value = [a*b for a,b in zip(sub_img_flat,gaussian_kernal)]
            a[i][j] = np.sum(value)/np.sum(gaussian_kernal)
    return a


# reading a first image file in black and white mode

print("Reading and resizing the first image")
img1 = cv2.imread(path_of_image1,0)

# resizing image to 200 X 400 size image

img1 = cv2.resize(img1, (400,200) )

# displaying all the image properties

print("Shape of First Image: ", img1.shape)
print("Minimum Intensity: ", img1.min())
print("Maximum Intensity: ", img1.max())
if(show_images_flag):
    plt.imshow(img1)
    plt.title("First Image")
    plt.show()

# reading a second image file in black and white mode

print("Reading and resizing the second image")
img2 = cv2.imread(path_of_image2,0)

# resizing image to 200 X 400 size image

img2 = cv2.resize(img2, (400,200) )

# displaying all the image properties

print("Shape of Second Image: ", img2.shape)
print("Minimum Intensity: ", img2.min())
print("Maximum Intensity: ", img2.max())

if(show_images_flag):
    plt.imshow(img2)
    plt.title("Second Image")
    plt.show()

####### Preparing image for performing hour transform on them #######
####### so that we will be able to divide two sides of the road #######
####### and then look into optical flow.  I will do an hough transform #######
####### on only one image as it is enough to determine the boundaries #######
####### of the left and right road  Assumption: there is no drastic #######
####### movement of camera in between both the image #######

print("Preparing image for Hough Transform")

# smoothing
print("Smoothing the image")
smooth_image1 = smooth_image(img1,3)

# detecting edges
print("Doing edge detection")
edge_output1, orientation_output1 = edge_detection(smooth_image1)

# thresholding
image1 = threshold(edge_output1, 0)

if(show_images_flag):
    print("Output after edge detection")
    # displaying smoothed, edge detected and thresholded image
    plt.imshow(image1)
    plt.title("Smoothed, Edge Detected and Thresholded Image1")
    plt.show()

# creating a histogram to analyze about the thresholded image
hist = histogram(image1)

# print("Computing histogram of edge detected image")
# #visualizing the histogram
# if(show_images_flag):
#     y_hist = np.arange(len(hist))
#     plt.bar(y_hist, hist)
#     plt.title("Histogram of Thresholded Image1")
#     plt.show()

#plotting the edges on hough space
start = time.time()
print("Starting hough transform")

large_dim = int(math.sqrt(math.pow(image1.shape[0],2)+math.pow(image1.shape[1],2)))
accumulator = np.zeros((large_dim, 180))
for i in range(0, image1.shape[0]):
    for j in range(0, image1.shape[1]):
        if(image1[i][j] > 0):
            for k in range(0, 180):
                r = (i*np.cos(np.deg2rad(k))) + (j*np.sin(np.deg2rad(k)))
                # considering in weight of edges as incremental factor in hough space accumulator
                accumulator[int(np.round(r))][k] = accumulator[int(np.round(r))][k] + (image1[i][j]/255.0)


end = time.time()
print("time taken for hough transform", end - start) # usually takes around 2.5 to 3 minutes for 200 X 400 image size

# displaying accumulator
if(show_images_flag):
    print("Plotting Hough Space")
    accumulator_display = change_range(accumulator)
    plt.imshow(accumulator_display)
    plt.title("Hough space for Image 1 after edge detection and thresholding")
    plt.show()

# finding local maximum peaks in accumulator

print("Finding the local maximuma and thresholded peaks")

r_list = []
theta_list = []

####### 2 ways of finding peaks and I am using both the ways #######
####### First Way: find the pixel which has more intensity pixel than all #######
####### of its 8N neighbourhood #######
####### Second Way: thresholding (I find this to be very useful) #######

for i in range(1, accumulator.shape[0]-1):
    for j in range(1, accumulator.shape[1]-1):
        if(accumulator[i][j]>accumulator[i-1][j] and
          accumulator[i][j]>accumulator[i-1][j+1] and
          accumulator[i][j]>accumulator[i-1][j-1] and
          accumulator[i][j]>accumulator[i+1][j] and
          accumulator[i][j]>accumulator[i+1][j-1] and
          accumulator[i][j]>accumulator[i+1][j+1] and
          accumulator[i][j]>accumulator[i][j-1] and
          accumulator[i][j]>accumulator[i][j+1] and
          accumulator[i][j]>50):
            r_list.append(i)
            theta_list.append(j)

print("Number of lines found", len(r_list))
print("Rho List", r_list)
print("Theta List", theta_list)

# filtering required middle dividers and filter them

print("Filtering only required middle lines and finding the best divider line")
req_theta_list = []
req_r_list = []
for i in range(0, len(theta_list)):
    if(theta_list[i] > 87 and theta_list[i] < 94 and r_list[i] > ((image1.shape[1]/2)-25) and r_list[i] < ((image1.shape[1]/2)+25) ):
        req_r_list.append(r_list[i])
        req_theta_list.append(theta_list[i])

print(req_r_list, req_theta_list)
right_theta_angle = round(sum(req_theta_list)/len(req_theta_list))
right_r_distance = round(sum(req_r_list)/len(req_r_list))
print(right_r_distance, right_theta_angle)

divider_column = right_r_distance

#displaying those lines on an empty image

r_list = [right_r_distance]
theta_list = [right_theta_angle]

lines_image = np.zeros((image1.shape[0],image1.shape[1]))
for k in range(0, len(r_list)):
    for i in range(1, image1.shape[0]):
        x = i
        y = ((-np.cos(np.deg2rad(theta_list[k]))/np.sin(np.deg2rad(theta_list[k])))*x) + ((r_list[k])/np.sin(np.deg2rad(theta_list[k])))
        if(math.isnan(y)):
            lines_image[x][x] = 255
            continue
        y = int(np.round(y))
        if(y>=0 and y< image1.shape[1]):
            lines_image[x][y] = 255

if(show_images_flag):
    plt.imshow(lines_image)
    plt.title("The DIVIDER LINE plotted on an empty image")
    plt.show()

####### Reality: We should place our camera somewhere in the middle of the #######
####### road and display a red light (Our main goal) #######
####### Corner Case: the road can be curved #######
####### (We do not look into such cases, thats a different study altogether) #######

# To make masks
# Left Mask can be used for vehicles driving on the left side of the road
# Right Mask can be used for vehicles driving on the right side of the road

print("Implementing the right and left maskes")
maskl = np.zeros((img1.shape[0], img1.shape[1]))
for i in range(0, lines_image.shape[0]):
    for j in range(0, lines_image.shape[1]):
        if(lines_image[i][j] == 255):
            break
        else:
            maskl[i][j] = 1

maskr = np.ones((img1.shape[0], img1.shape[1]))
for i in range(0, lines_image.shape[0]):
    for j in range(0, lines_image.shape[1]):
        if(lines_image[i][j] == 255):
            break
        else:
            maskr[i][j] = 0

if(show_images_flag):
    plt.imshow(maskr)
    plt.title("Right Mask")
    plt.show()

    plt.imshow(maskl)
    plt.title("Left Mask")
    plt.show()

# Analyzing left side of the road

# Applying Left Mask to first and second images
print("Applying Left Mask to first and second images")
img1l = img1*maskl
img2l = img2*maskl

if(show_images_flag):
    plt.imshow(img1l)
    plt.title("Image 1 with Left Mask")
    plt.show()

    plt.imshow(img2l)
    plt.title("Image 2 with Left Mask")
    plt.show()

# Applying Right Mask to first and second images
print("Applying Right Mask to first and second images")
img1r = img1*maskr
img2r = img2*maskr

if(show_images_flag):
    plt.imshow(img1r)
    plt.title("Image 1 with Right Mask")
    plt.show()

    plt.imshow(img2r)
    plt.title("Image 2 with Right Mask")
    plt.show()

####### OPTICAL FLOW starts from here using Lucas Kalade technique #######
####### care has been taken that the u and v displacement between frames is very small #######


print("Applying Guassian bluring on all the four images")
# img1l = cv2.GaussianBlur(img1l,(3,3),2)
img1l = GaussianBlur_custom(img1l, 3, 2)
if(show_images_flag):
    plt.imshow(img1l)
    plt.title("Gaussian Blurred 1st Left Image")
    plt.show()

# img2l = cv2.GaussianBlur(img2l,(3,3),2)
img2l = GaussianBlur_custom(img2l, 3, 2)
if(show_images_flag):
    plt.imshow(img2l)
    plt.title("Gaussian Blurred 2nd Left Image")
    plt.show()

# img1r = cv2.GaussianBlur(img1r,(3,3),2)
img1r = GaussianBlur_custom(img1r, 3, 2)
if(show_images_flag):
    plt.imshow(img1r)
    plt.title("Gaussian Blurred 1st Right Image")
    plt.show()

# img2r = cv2.GaussianBlur(img2r,(3,3),2)
img2r = GaussianBlur_custom(img2r, 3, 2)
if(show_images_flag):
    plt.imshow(img2r)
    plt.title("Gaussian Blurred 2nd Right Image")
    plt.show()

####### finding x gradient, y gradient and time gradient for both the frames #######
####### I am finding gradients for both the images and averaging it #######
####### as suggested by one of the professors in UCF #######

print("Applying LUCAS KANADE OPTICAL FLOW on both the images on both the sides of the road")

print("Applying derivative averaging for both the images on right and left sides using loberts masks")

start_time = time.time()

zero_img_matrix = np.zeros(img1.size,float).reshape(img1.shape[0],img1.shape[1])
x_gradient_img1l = zero_img_matrix.copy()
y_gradient_img1l = zero_img_matrix.copy()
x_gradient_img2l = zero_img_matrix.copy()
y_gradient_img2l = zero_img_matrix.copy()
x_gradient_imgl = zero_img_matrix.copy()
y_gradient_imgl = zero_img_matrix.copy()
t_gradient_img1l = zero_img_matrix.copy()
t_gradient_img2l = zero_img_matrix.copy()
t_gradient_imgl = zero_img_matrix.copy()

for i in range(0, img1.shape[0]-2):
    for j in range(0, img1.shape[1]-2):
        x_gradient_img1l[i][j] = (-1*(img1l[i][j])) + (-1*(img1l[i+1][j])) + (1*(img1l[i][j+1])) + (1*(img1l[i+1][j+1]))
        x_gradient_img2l[i][j] = (-1*(img2l[i][j])) + (-1*(img2l[i+1][j])) + (1*(img2l[i][j+1])) + (1*(img2l[i+1][j+1]))
        y_gradient_img1l[i][j] = (-1*(img1l[i][j])) + (1*(img1l[i+1][j])) + (-1*(img1l[i][j+1])) + (1*(img1l[i+1][j+1]))
        y_gradient_img2l[i][j] = (-1*(img2l[i][j])) + (1*(img2l[i+1][j])) + (-1*(img2l[i][j+1])) + (1*(img2l[i+1][j+1]))
        t_gradient_img1l[i][j] = (-1*(img1l[i][j])) + (-1*(img1l[i+1][j])) + (-1*(img1l[i][j+1])) + (-1*(img1l[i+1][j+1]))
        t_gradient_img2l[i][j] = (+1*(img2l[i][j])) + (+1*(img2l[i+1][j])) + (+1*(img2l[i][j+1])) + (+1*(img2l[i+1][j+1]))
        x_gradient_imgl = x_gradient_img1l + x_gradient_img2l
        y_gradient_imgl = y_gradient_img1l + y_gradient_img2l
        t_gradient_imgl = t_gradient_img1l + t_gradient_img2l



# x_gradient_imgl = cv2.GaussianBlur(x_gradient_imgl,(3,3),2)
x_gradient_imgl = GaussianBlur_custom(x_gradient_imgl, 3, 2)
# y_gradient_imgl = cv2.GaussianBlur(y_gradient_imgl,(3,3),2)
y_gradient_imgl = GaussianBlur_custom(y_gradient_imgl, 3, 2)
# t_gradient_imgl = cv2.GaussianBlur(t_gradient_imgl,(3,3),2)
t_gradient_imgl = GaussianBlur_custom(t_gradient_imgl, 3, 2)

x_gradient_img1r = zero_img_matrix.copy()
y_gradient_img1r = zero_img_matrix.copy()
x_gradient_img2r = zero_img_matrix.copy()
y_gradient_img2r = zero_img_matrix.copy()
x_gradient_imgr = zero_img_matrix.copy()
y_gradient_imgr = zero_img_matrix.copy()
t_gradient_img1r = zero_img_matrix.copy()
t_gradient_img2r = zero_img_matrix.copy()
t_gradient_imgr = zero_img_matrix.copy()

for i in range(0, img1.shape[0]-2):
    for j in range(0, img1.shape[1]-2):
        x_gradient_img1r[i][j] = (-1*(img1r[i][j])) + (-1*(img1r[i+1][j])) + (1*(img1r[i][j+1])) + (1*(img1r[i+1][j+1]))
        x_gradient_img2r[i][j] = (-1*(img2r[i][j])) + (-1*(img2r[i+1][j])) + (1*(img2r[i][j+1])) + (1*(img2r[i+1][j+1]))
        y_gradient_img1r[i][j] = (-1*(img1r[i][j])) + (1*(img1r[i+1][j])) + (-1*(img1r[i][j+1])) + (1*(img1r[i+1][j+1]))
        y_gradient_img2r[i][j] = (-1*(img2r[i][j])) + (1*(img2r[i+1][j])) + (-1*(img2r[i][j+1])) + (1*(img2r[i+1][j+1]))
        t_gradient_img1r[i][j] = (-1*(img1r[i][j])) + (-1*(img1r[i+1][j])) + (-1*(img1r[i][j+1])) + (-1*(img1r[i+1][j+1]))
        t_gradient_img2r[i][j] = (+1*(img2r[i][j])) + (+1*(img2r[i+1][j])) + (+1*(img2r[i][j+1])) + (+1*(img2r[i+1][j+1]))
        x_gradient_imgr = x_gradient_img1r + x_gradient_img2r
        y_gradient_imgr = y_gradient_img1r + y_gradient_img2r
        t_gradient_imgr = t_gradient_img1r + t_gradient_img2r


# x_gradient_imgr = cv2.GaussianBlur(x_gradient_imgr,(3,3),2)
x_gradient_imgr = GaussianBlur_custom(x_gradient_imgr, 3, 2)
# y_gradient_imgr = cv2.GaussianBlur(y_gradient_imgr,(3,3),2)
y_gradient_imgr = GaussianBlur_custom(y_gradient_imgr, 3, 2)
# t_gradient_imgr = cv2.GaussianBlur(t_gradient_imgr,(3,3),2)
t_gradient_imgr = GaussianBlur_custom(t_gradient_imgr, 3, 2)

end_time = time.time()

print("Time taken to apply gradient averaging and applying gaussian blur", end_time - start_time)

####### Displaying X, Y and time gradient graphs #######

if(show_images_flag):
    print("Showing averaged X, Y and time gradient of left and right images")

    plt.imshow(change_range(x_gradient_imgl))
    plt.title("Averaged X gradient of Left images")
    plt.show()
    plt.imshow(change_range(y_gradient_imgl))
    plt.title("Averaged Y gradient of Left images")
    plt.show()
    plt.imshow(change_range(t_gradient_imgl))
    plt.title("Averaged Time gradient of Left images")
    plt.show()

    plt.imshow(change_range(x_gradient_imgr))
    plt.title("Averaged X gradient of Right images")
    plt.show()
    plt.imshow(change_range(y_gradient_imgr))
    plt.title("Averaged Y gradient of Right images")
    plt.show()
    plt.imshow(change_range(t_gradient_imgr))
    plt.title("Averaged Time gradient of Right images")
    plt.show()

####### compute u and v using lucas kanade method, i am not using matrix inverse #######
####### I am using the least mean squares

start_time = time.time()
print("Determining U and V matrices for the left side of the road images")

####### For left side of the road #######

ul = zero_img_matrix.copy()
vl = zero_img_matrix.copy()
uv_magl = zero_img_matrix.copy()
uv_dirl = zero_img_matrix.copy()

for i in range(1, img1.shape[0]-2):
    for j in range(1, divider_column):
        if(maskl[i][j] == 1):
                fx = [x_gradient_imgl[i-1][j-1],x_gradient_imgl[i-1][j],x_gradient_imgl[i-1][j+1],
                 x_gradient_imgl[i][j-1],x_gradient_imgl[i][j],x_gradient_imgl[i][j+1],
                 x_gradient_imgl[i+1][j-1],x_gradient_imgl[i+1][j],x_gradient_imgl[i+1][j+1]]
                fy = [y_gradient_imgl[i-1][j-1],y_gradient_imgl[i-1][j],y_gradient_imgl[i-1][j+1],
                     y_gradient_imgl[i][j-1],y_gradient_imgl[i][j],y_gradient_imgl[i][j+1],
                     y_gradient_imgl[i+1][j-1],y_gradient_imgl[i+1][j],y_gradient_imgl[i+1][j+1]]
                ft = [t_gradient_imgl[i-1][j-1],t_gradient_imgl[i-1][j],t_gradient_imgl[i-1][j+1],
                     t_gradient_imgl[i][j-1],t_gradient_imgl[i][j],t_gradient_imgl[i][j+1],
                     t_gradient_imgl[i+1][j-1],t_gradient_imgl[i+1][j],t_gradient_imgl[i+1][j+1]]

                sum_fx2 = np.sum(np.square(fx))
                sum_fy2 = np.sum(np.square(fy))
                sum_fxy = np.sum([a*b for a,b in zip(fx,fy)])
                sum_fxy_2 = np.square(sum_fxy)

                denom = sum_fx2*sum_fy2-sum_fxy_2
                if(denom == 0.0):
                    print(i,j)
                    continue


                sum_fxt = np.sum([a*b for a,b in zip(fx,ft)])
                sum_fyt = np.sum([a*b for a,b in zip(fy,ft)])

                ul[i][j] = ((-sum_fy2*sum_fxt) + (sum_fxy*sum_fyt))/denom
                vl[i][j] = ((sum_fxt*sum_fxy) - (sum_fx2*sum_fyt))/denom

print("Applying median filter on left u and v")
ul = medianfilter(ul,3)
vl = medianfilter(vl,3)

for i in range(1, img1.shape[0]-2):
    for j in range(1, divider_column):
        uv_magl[i][j] = math.sqrt((ul[i][j]*ul[i][j])+(vl[i][j]*vl[i][j]))
        uv_dirl[i][j] = math.degrees(math.atan(ul[i][j]/(vl[i][j]+0.0001)))

print("Determining U and V matrices for the right side of the road images")
####### For right side of the road #######


ur = zero_img_matrix.copy()
vr = zero_img_matrix.copy()
uv_magr = zero_img_matrix.copy()
uv_dirr = zero_img_matrix.copy()

for i in range(1, img1.shape[0] - 2):
    for j in range(divider_column, img1.shape[1] - 2):
        if(maskr[i][j] == 1):
            fx = [x_gradient_imgr[i-1][j-1],x_gradient_imgr[i-1][j],x_gradient_imgr[i-1][j+1],
                 x_gradient_imgr[i][j-1],x_gradient_imgr[i][j],x_gradient_imgr[i][j+1],
                 x_gradient_imgr[i+1][j-1],x_gradient_imgr[i+1][j],x_gradient_imgr[i+1][j+1]]
            fy = [y_gradient_imgr[i-1][j-1],y_gradient_imgr[i-1][j],y_gradient_imgr[i-1][j+1],
                 y_gradient_imgr[i][j-1],y_gradient_imgr[i][j],y_gradient_imgr[i][j+1],
                 y_gradient_imgr[i+1][j-1],y_gradient_imgr[i+1][j],y_gradient_imgr[i+1][j+1]]
            ft = [t_gradient_imgr[i-1][j-1],t_gradient_imgr[i-1][j],t_gradient_imgr[i-1][j+1],
                 t_gradient_imgr[i][j-1],t_gradient_imgr[i][j],t_gradient_imgr[i][j+1],
                 t_gradient_imgr[i+1][j-1],t_gradient_imgr[i+1][j],t_gradient_imgr[i+1][j+1]]

            sum_fx2 = np.sum(np.square(fx))
            sum_fy2 = np.sum(np.square(fy))
            sum_fxy = np.sum([a*b for a,b in zip(fx,fy)])
            sum_fxy_2 = np.square(sum_fxy)

            denom = sum_fx2*sum_fy2-sum_fxy_2
            if(denom == 0.0):
                print(i,j)
                continue


            sum_fxt = np.sum([a*b for a,b in zip(fx,ft)])
            sum_fyt = np.sum([a*b for a,b in zip(fy,ft)])

            ur[i][j] = ((-sum_fy2*sum_fxt) + (sum_fxy*sum_fyt))/denom
            vr[i][j] = ((sum_fxt*sum_fxy) - (sum_fx2*sum_fyt))/denom

print("Applying median filter on right u and v")
ur = medianfilter(ur,3)
vr = medianfilter(vr,3)

for i in range(1, img1.shape[0]-2):
    for j in range(divider_column, img1.shape[1]-2):
        uv_magr[i][j] = math.sqrt((ur[i][j]*ur[i][j])+(vr[i][j]*vr[i][j]))
        uv_dirr[i][j] = math.degrees(math.atan(ur[i][j]/(vr[i][j]+0.0001)))

end_time = time.time()

print("Complete time taken to find u and v matrices for both the sides of the road and apply median filter on it", end_time - start_time)


####### plotting uv direction and magnitude graph of the left side of the road #######
if(show_images_flag):
    print("Showing the UV direction, magnitude of both left and right side images of the road")
    plt.imshow(change_range(uv_dirl))
    plt.title("UV direction of left side of the road")
    plt.show()

    plt.imshow(change_range(uv_magl))
    plt.title("UV Magnitute of left side of the road")
    plt.show()

    plt.imshow(change_range(uv_dirr))
    plt.title("UV direction of right side of the road")
    plt.show()

    plt.imshow(change_range(uv_magr))
    plt.title("UV Magnitute of right side of the road")
    plt.show()

####### looking at the left side of the road analysis of UV direction #######

magnitude_threshold = 10
print("Analyzing the histogram of the UV directions of the left side of the road")
angles_list = []
for i in range(0, img1.shape[0]):
    for j in range(0, divider_column):
        if(uv_magl[i][j]> magnitude_threshold):
            angles_list.append(uv_dirl[i][j])


data = angles_list

# fixed bin size
bins = np.arange(-200, 200, 5) # fixed bin size

if(show_images_flag):

    plt.xlim([min(data)-5, max(data)+5])

    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('UV orientation of left side of the bin (fixed bin size)')
    plt.xlabel('UV orientation X (bin size = 5)')
    plt.ylabel('count')
    plt.show()

####### Determining the degree of the vehicle on the left side of the vehicle #######
bins_for_histogram = 5

max_x = 90 + bins_for_histogram
min_x = -90 + bins_for_histogram

bin_list = []
for i in range(min_x, max_x, 5):
    bin_list.append(i)

countprev = 0
value_list = []
for i in range(len(bin_list)):
    count = 0
    for j in range(len(data)):

        if(data[j] < bin_list[i]):
            count += 1
    value_list.append(count - countprev)
    countprev = count

# weighted_sum_list = [a*b for a,b in zip(bin_list,value_list)]
# weighted_angle_value_left = sum(weighted_sum_list)/sum(value_list)
# max_degree_left = weighted_angle_value_left

max_index, max_value = max(enumerate(value_list), key=operator.itemgetter(1))

max_degree_left = bin_list[max_index]


####### Right side of the road analaysis #######
print("Analyzing the histogram of the UV directions of the right side of the road")

angles_list = []
for i in range(0, img1.shape[0]):
    for j in range(divider_column, img1.shape[1]):
        if(uv_magr[i][j]> magnitude_threshold):
            angles_list.append(uv_dirr[i][j])


data = angles_list

# fixed bin size
bins = np.arange(-200, 200, 5) # fixed bin size

if(show_images_flag):
    plt.xlim([min(data)-5, max(data)+5])

    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('UV orientation of right side of the bin (fixed bin size)')
    plt.xlabel('UV orientation X (bin size = 5)')
    plt.ylabel('count')
    plt.show()

####### Determining the degree of the vehicle on the left side of the vehicle #######
bins_for_histogram = 1

max_x = 90 + bins_for_histogram
min_x = -90 + bins_for_histogram

bin_list = []
for i in range(min_x, max_x, 5):
    bin_list.append(i)

countprev = 0
value_list = []
for i in range(len(bin_list)):
    count = 0
    for j in range(len(data)):

        if(data[j] < bin_list[i]):
            count += 1
    value_list.append(count - countprev)
    countprev = count

# weighted_sum_list = [a*b for a,b in zip(bin_list,value_list)]
# weighted_angle_value_right = sum(weighted_sum_list)/sum(value_list)
# max_degree_right = weighted_angle_value_right

max_index, max_value = max(enumerate(value_list), key=operator.itemgetter(1))

max_degree_right = bin_list[max_index]

print("FINAL RESULTS")
left_correct = False
right_correct = False
if(max_degree_left < 0):
    print("The vehicle on the LEFT SIDE OF THE ROAD IS IN THE CORRECT DIRECTION")
    left_correct = True
else:
    print("The vehicle on the LEFT SIDE OF THE ROAD IS IN THE WRONG DIRECTION")

if(max_degree_right > 0):
    print("The vehicle on the RIGHT SIDE OF THE ROAD IS IN THE CORRECT DIRECTION")
    right_correct = True
else:
    print("The vehicle on the RIGHT SIDE OF THE ROAD IS IN THE WRONG DIRECTION")
