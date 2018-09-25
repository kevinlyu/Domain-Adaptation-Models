import os
import numpy as numpy
from skimage import io, transform
import numpy as np

weight = 200
hight = 200
data = None
labels = None

domains = ["amazon", "dslr", "webcam"]

def process_image(image, weight, hight):
    img = io.imread(image)
    img = transform.resize(img, (weight, hight), mode="reflect")
    return img


for d in domains:
    path = "../dataset/office31/" + d + "/"
    print("processing " + path)

    for _, dirnames, _ in os.walk(path):
        dirnames.sort()
        for dirname in dirnames:
            index = dirnames.index(dirname)
            workdir = os.path.join(path, dirname)
            print(workdir)
            processed_images = io.ImageCollection(
                workdir + "/*.jpg", load_func=process_image, weight=weight, hight=hight)
            label = np.full(len(processed_images),
                            fill_value=index, dtype=np.int32)
            images = io.concatenate_images(processed_images)

            if index == 0:
                data = images
                labels = label

            else:
                data = np.vstack((data, images))
                labels = np.append(labels, label)

    print(np.shape(data))
    print(np.shape(labels))

    np.savez("../dataset/office31/"+d+"31.npz", data=data, label=labels)
    print("saved " + d + "31.npz")
