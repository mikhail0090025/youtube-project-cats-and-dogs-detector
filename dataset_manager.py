from PIL import Image
import numpy as np
import os

cats_path = os.path.join("PetImages", "Cat")
dogs_path = os.path.join("PetImages", "Dog")

max_photos_per_folder = 10000
image_size = (100, 100)
all_dogs = []
all_cats = []

def get_cats():
    global all_cats
    for file in os.listdir(cats_path)[:max_photos_per_folder]:
        img = Image.open(os.path.join(cats_path, file)).convert("RGB").resize(image_size)
        img_numpy = np.array(img, dtype=np.float32)
        img_numpy /= 127.5
        img_numpy -= 1
        all_cats.append(img_numpy)
    all_cats = np.array(all_cats)
    all_cats = np.transpose(all_cats, (0, 3, 1, 2))

def get_dogs():
    global all_dogs
    for file in os.listdir(dogs_path)[:max_photos_per_folder]:
        img = Image.open(os.path.join(dogs_path, file)).convert("RGB").resize(image_size)
        img_numpy = np.array(img, dtype=np.float32)
        img_numpy /= 127.5
        img_numpy -= 1
        all_dogs.append(img_numpy)
    all_dogs = np.array(all_dogs)
    all_dogs = np.transpose(all_dogs, (0, 3, 1, 2))


def save():
    np.savez_compressed("dogs.npz", images=all_dogs)
    np.savez_compressed("cats.npz", images=all_cats)

def load():
    global all_dogs, all_cats
    all_dogs = np.load("dogs.npz")['images']
    all_cats = np.load("cats.npz")['images']

if os.path.exists("dogs.npz") and os.path.exists("cats.npz"):
    load()
else:
    get_cats()
    get_dogs()
    save()

print(all_dogs.shape)
print(all_cats.shape)