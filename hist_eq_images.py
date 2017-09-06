import pickle
import matplotlib.pyplot as plt
import numpy as np
import preprocess_data as pp

imgs_file = "../data/traffic-signs-data/test.p"

with open(imgs_file, mode='rb') as f:
    imgs = pickle.load(f)

X, y = imgs['features'], imgs['labels']
print('X dtype, y dtype', X.dtype, y.dtype)

X_eq = pp.hist_eq(X)

print("Pickling file")
# write hist_eq pickle file
imgs_hist_eq_file = "../data/traffic-signs-data/test_hist_eq.p"

imgs_hist_eq_data = {'features': X_eq, 'labels': y}
with open(imgs_hist_eq_file, mode='wb') as f:
    pickle.dump(imgs_hist_eq_data, f)

plt.imshow(X[-1,...])
plt.show()
plt.imshow(X_eq[-1,...])
plt.show()