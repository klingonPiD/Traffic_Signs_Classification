"""Script to augment the given training data"""

# Load pickled data
import pickle
import matplotlib.pyplot as plt
import numpy as np
from transform_image import *
from sklearn.model_selection import train_test_split

training_file = "../data/traffic-signs-data/train.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']
print('X_train dtype, y_train dtype', X_train.dtype, y_train.dtype)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.2)

print("X_train shape is", X_train.shape)
print("X_valid shape is", X_valid.shape)

n_train = len(set(y_train))
hist, _ = np.histogram(y_train, n_train)
#print(hist[0], hist[10])
#print(hist[0], hist[10])
X_aug, y_aug = np.array([]),np.array([])

count = 0
for i in range(n_train):
    print("processing tr sign", i)
    n_samp = hist[i]
    if n_samp < 600: #600#750#1500
        delta = 600 - n_samp
        # add delta amount of affine transforms to random images of this type
        indices = np.where(y_train == i)[0]
        for samp in range(delta):
            index = np.random.choice(indices)
            #out_image = np.empty(shape=(1, PIXELS, PIXELS, 3), dtype='float32')
            out_image = transform_image(X_train[index,...])
            if X_aug.size == 0:
                X_aug  = out_image.astype(np.uint8)
                y_aug = np.array([i]).astype(np.uint8)
            else:
                X_aug = np.concatenate((X_aug, out_image.astype(np.uint8)))
                y_aug = np.concatenate((y_aug,np.array([i]).astype(np.uint8)))
    if n_samp > 1250: #1500
        delta = n_samp - 1250
        indices = np.where(y_train == i)[0]
        X_train = np.delete(X_train, indices[0:delta], axis=0).astype(np.uint8)
        y_train = np.delete(y_train, indices[0:delta], axis=0).astype(np.uint8)

print("n_samp, delta", n_samp, delta)
print("X_train shape is", X_train.shape)

print("Final concatenation")
#do final concatenation
X_aug = np.concatenate((X_train, X_aug))
y_aug = np.concatenate((y_train, y_aug))

print("X_aug, y_aug shape", X_aug.shape, y_aug.shape)

print("Pickling file")
# write augmented pickle file
train_aug_file = "../data/traffic-signs-data/train_aug.p"
valid_aug_file = "../data/traffic-signs-data/valid_aug.p"

train_aug_data = {'features': X_aug, 'labels': y_aug}
with open(train_aug_file, mode='wb') as f:
    pickle.dump(train_aug_data, f)

valid_aug_data = {'features': X_valid, 'labels': y_valid}
with open(valid_aug_file, mode='wb') as f:
    pickle.dump(valid_aug_data, f)

plt.imshow(X_aug[-1,...])
plt.show()

