import pickle

import cv2
from imutils import paths
import numpy as np
from keras.models import load_model

from preprocess import make_counters, resize_symbol

model_file = "model.joblib"
labels_file = "labels.dat"
test_images_folder = "test"

with open(labels_file, "rb") as f:
    lb = pickle.load(f)

model = load_model(model_file)

test_images = list(paths.list_images(test_images_folder))
test_images = np.random.choice(test_images, size=(10,), replace=False)

for image_file in test_images:
    image = cv2.imread(image_file)
    output = image.copy()

    contour_regions, gray = make_counters(image)
    predictions = []
    for symbol_box in contour_regions:
        x, y, w, h = symbol_box
        symbol_image = gray[y: y + h, x: x + w]
        symbol_image = resize_symbol(symbol_image, 28)

        symbol_image = np.expand_dims(symbol_image, axis=0)
        symbol_image = np.expand_dims(symbol_image, axis=3)

        prediction = model.predict(symbol_image)

        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))

    cv2.imshow("Output", output)
    cv2.waitKey()
