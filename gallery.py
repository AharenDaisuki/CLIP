import random

from matplotlib import pyplot as plt

def show_winoground(winoground): 
    index = random.randint(0, len(winoground) - 1)
    ax1 = plt.subplot(2, 1, 1)
    ax1.title.set_text('image_0')
    plt.imshow(winoground[index]["image_0"].convert("RGB"))

    ax2 = plt.subplot(2, 1, 2)
    ax2.title.set_text('image_1')
    plt.imshow(winoground[index]["image_1"].convert("RGB"))

    plt.show()

    print("caption_0:", winoground[index]["caption_0"])
    print("caption_1:", winoground[index]["caption_1"])

def show_flickr(flickr): 
    index = random.randint(0, len(flickr) - 1)
    ax1 = plt.subplot(2, 1, 1)
    ax1.title.set_text('image')
    plt.imshow(flickr[index]["image"].convert("RGB"))

    # ax2 = plt.subplot(2, 1, 2)
    # ax2.title.set_text('image_1')
    # plt.imshow(flickr[index]["image_1"].convert("RGB"))

    plt.show()

    print("caption_0:", flickr[index]["caption_0"])
    print("caption_1:", flickr[index]["caption_1"])
    print("caption_0:", flickr[index]["caption_2"])
    print("caption_1:", flickr[index]["caption_3"])
    print("caption_0:", flickr[index]["caption_4"])