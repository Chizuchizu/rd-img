import numpy as np
import cv2
import matplotlib.pyplot as plt

is_plt = True
# plt.imshow(array)
# plt.show()

def img2binary(img):
        array = img.flatten()
        ar = array.copy()

        barray = np.zeros((300, 8)).astype(np.uint8)
        for i in range(8):
            barray[:, i] = ar % 2
            ar //= 2

        return barray.flatten()

def transform_encode(img, binary):
        img_array = img.flatten()
        img_array //= 2
        img_array *= 2
        img_array += binary

        return img_array.reshape(img.shape)


def decode_img(img):
        binary = img.flatten() % 2
        binary = binary.reshape(10, 10, 3, 8)

        new_img = np.zeros((10, 10, 3), dtype=np.uint8)

        for i in range(8):
                new_img += binary[:, :, :, i] * (2 ** i)

        return new_img


private_img = cv2.imread("1010.png")
barray = img2binary(private_img)


img = cv2.imread("1080.png")

# plt.imshow(img)
img_encoded = transform_encode(img, barray)

new_img = decode_img(img_encoded)

# """
if is_plt:
        img_encoded = np.where(img_encoded == 254, 0, 255)
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("private img")
        plt.imshow(private_img)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("decoded img")
        plt.imshow(new_img)
        plt.show()

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title("original img")
        plt.imshow(img)
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_title("encoded img")
        plt.imshow(img_encoded)
        plt.show()
        # """

print(np.unique(img_encoded))
