import torchvision
import matplotlib.pyplot as plt

'''
Show reconstructed images.
'''
def show_images(images):
    images = torchvision.utils.make_grid(images)
    show_image(images)


def show_image(img):
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.show()




