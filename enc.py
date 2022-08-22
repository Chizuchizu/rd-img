import numpy as np
import cv2
import matplotlib.pyplot as plt

from mk_img import make_random_img, make_white_img

is_plt = True
coefficient = 1  # default 1, 画像の変化を見たいときに使う

def img2binary(img, size):
        array = img.flatten()
        ar = array.copy()

        barray = np.zeros((size, 8)).astype(np.uint8)
        for i in range(8):
            barray[:, i] = ar % 2
            ar //= 2

        return barray.flatten()

def transform_encode(img, binary):
        img_array = img.flatten()

        output_length = len(img_array)
        hidden_length = len(binary)

        if output_length < hidden_length:
                raise
        elif output_length > hidden_length:
                binary = np.pad(binary, [0, output_length - hidden_length], "constant")

        img_array //= 2
        img_array *= 2
        img_array += binary * coefficient

        return img_array.reshape(img.shape)


def decode_img(img, size, shape):
        binary = img.flatten() % 2
        binary = binary[:size * 8]
        binary = binary.reshape(shape + (8,))

        new_img = np.zeros(shape, dtype=np.uint8)

        for i in range(8):
                new_img += binary[:, :, :, i] * (2 ** i)

        return new_img


# private_img = cv2.imread("1010.png")
# private_img = make_random_img(10, 10)
private_img = cv2.imread("../img/icon_face.png")
private_img =  cv2.cvtColor(private_img, cv2.COLOR_BGR2RGB)
barray = img2binary(private_img, size=private_img.size)


# img = cv2.imread("1080.png")
# img = make_white_img(400, 800)
img = cv2.imread("../img/room.jpeg")
img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_encoded = transform_encode(img, barray)
new_img = decode_img(img_encoded, size=private_img.size, shape=private_img.shape)

# """
if is_plt:
        # img_encoded = np.where(img_encoded == 254, 0, 255)
        # """
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("private img")
        plt.imshow(private_img)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("decoded img")
        plt.imshow(new_img)
        plt.show()
        # """
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title("original img")
        plt.imshow(img)
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_title("encoded img")
        plt.imshow(img_encoded)
        plt.show()
        # """

