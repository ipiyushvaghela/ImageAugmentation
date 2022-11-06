# Aim of this py file is create augmented image of given image randomly.

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import cv2 as cv
import glob

# Augment class which has diffrent function which can be applied on one image
class Augmentation:
    def __init__(self,img,resize_shape=(512,512),min_hight=200,min_width=200, angle=None, center = None, scale = None):
        self.img = img
        self.resize_shape = resize_shape
        self.min_hight = min_hight
        self.min_width = min_width
        self.angle = angle
        self.center =center
        self.scale = scale

    # Functions for Position augmentation
    def Resize_image(self):
        if self.img is None:
            print('Wrong path is Enterd')
        else:
            self.img = cv.resize(self.img,self.resize_shape)
            return self.img

    def Scaling(self):  #reduce the image height width and then resize image to normal shape so it will reduce image resolution
        image_shape = self.img.shape
        scalled_img = cv.resize(self.img,(random.choice(range(image_shape[0]//2,image_shape[0])),random.choice(range(image_shape[1]//2,image_shape[1]))))
        self.img = cv.resize(scalled_img,self.resize_shape)
        return self.img

    def Cropping(self):
        image_shape = self.img.shape

        upper_row = random.choice(range(0,image_shape[0]//4))
        lower_row = random.choice(range(upper_row + self.min_hight,image_shape[0]))

        start_column = random.choice(range(0,image_shape[1]//4))
        end_column = random.choice(range(start_column+self.min_width,image_shape[1]))

        crop = self.img[upper_row:lower_row, start_column:end_column]
        self.img = cv.resize(crop,self.resize_shape)
        return self.img

    def Flipping(self):
        self.img = cv.flip(self.img,random.choice((0,1,-1)))
        return self.img

    def Padding(self):
        image_shape = self.img.shape
        l = [cv.BORDER_CONSTANT,cv.BORDER_REFLECT,cv.BORDER_REFLECT_101,cv.BORDER_DEFAULT,cv.BORDER_REPLICATE,cv.BORDER_WRAP]
        img_with_padding = cv.copyMakeBorder(self.img, image_shape[0]//random.choice(range(5,16)), image_shape[0]//random.choice(range(5,16)), image_shape[1]//random.choice(range(5,16)), image_shape[1]//random.choice(range(5,16)),l[random.choice(range(0,6))])
        self.img = cv.resize(img_with_padding,self.resize_shape)
        return self.img

    # rotate image to a perticular angle
    def Rotation(self):
        (h, w) = self.img.shape[:2]

        if self.angle is None:
            self.angle = random.choice(range(0,361))

        if self.center is None:
            self.center = (w / 2, h / 2)

        self.scale = random.choice(np.linspace(1,2,9))
        # Perform the rotation
        M = cv.getRotationMatrix2D(self.center, self.angle, self.scale) # An affine transformation is transformation which preserves lines and parallelism.
        # These transformation matrix are taken by warpaffine() function as parameter and the rotated image will be returned.
        self.img = cv.warpAffine(self.img, M, (w, h))
        return self.img

    # Functions for Color augmentation
    # For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255].
    # Different software use different scales.
    # So if you are comparing OpenCV values with them, you need to normalize these ranges.


    def Brightness(self):
        value = random.randint(20,50)  # cause 255 sometimes is too much
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)

        if value>=0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = 0 - value
            v[v < lim] = 0
            v[v >= lim] -= value

        final_hsv = cv.merge((h, s, v))
        self.img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        return self.img

    def Saturation(self):
        value = random.randint(20,50) # cause 200 is sometime too much
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)

        if value>=0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = 0 - value
            s[s < lim] = 0
            s[s >= lim] -= value

        final_hsv = cv.merge((h, s, v))
        self.img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        return self.img

    def Hue(self):
        value = random.randint(20,50)
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)

        if value>=0:
            lim = 179 - value
            h[h > lim] = 179
            h[h <= lim] += value
        else:
            lim = 0 - value
            h[h < lim] = 0
            h[h >= lim] -= value

        final_hsv = cv.merge((h, s, v))
        self.img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        return self.img

    # contrast levels go from -127 to +127
    def Contrast(self):
        contrast = random.randint(-70,80)
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        self.img = cv.addWeighted(self.img, alpha_c, self.img, 0, gamma_c)

        return self.img

def random_augmented_image_generater(no_of_resulted_images,source,destination):
    ''' INPUT : no_of_resulted_images: How many augmented images should be generated per image in the source folder, 
                source : Source folder path
                destination: Destination folder path
        OUTPUT : nothing will be returned
        
        Use : will help you to generate augmented images of source folder and create n number of augmented images per source image 
        and store it to destination folder
    '''

    source += '\*.jpg'
    destination += "\\"
   
    files = glob.glob(source)

    for file in files:
        img = cv.imread(file)
        temp_no = 1

        for i in range(no_of_resulted_images):

            temp_img = img
            tooa = Augmentation(temp_img) # temp_object_of_Augmentation
            temp_img = tooa.Resize_image()

            method_list_of_position = [tooa.Scaling,tooa.Cropping,tooa.Flipping,tooa.Padding,tooa.Rotation]
            method_list_of_color = [tooa.Brightness,tooa.Hue,tooa.Saturation,tooa.Contrast]

            no_of_functios_to_apply = random.choice(range(2,len(method_list_of_position)+1))
            print(no_of_functios_to_apply)

            # Color Augmentation
            fun = random.choice(method_list_of_color)
            # print("function name is :" + fun.__name__) # to print function Name
            temp_img = fun()

            # Positional Augmentation
            for i in range(no_of_functios_to_apply-1):
                fun = random.choice(method_list_of_position)
                # print("function name is :" + fun.__name__) # to print function Name
                temp_img = fun()
                method_list_of_position.remove(fun)


            file_name = destination + file.split("\\")[-1].split('.')[0] +str(temp_no) + ".jpg"
            temp_no += 1
            cv.imwrite(file_name,temp_img)
        print("Your images are genrated and stored at : " + destination )
