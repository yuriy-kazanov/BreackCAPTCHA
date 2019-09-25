import cv2
import pickle
import numpy as np
from imutils import paths
from keras.models import load_model
from preprocess import color_to_black
from preprocess import clear_background
from preprocess import split_into_pieces
from preprocess import resize_symbol


model_file = "model.hdf5"
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
    image = cv2.imread(image_file)    
    
    image = color_to_black(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value = 255)
    height, width = gray.shape[:2]  
    gray = clear_background(gray)
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour_regions = split_into_pieces(contours, width, height)
    
    contour_regions = sorted(contour_regions, key=lambda x: x[0])
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
        
        
        
        
        
        
        
        
        