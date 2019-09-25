import os
import os.path
import cv2
import glob
from preprocess import color_to_black
from preprocess import clear_background
from preprocess import split_into_pieces
from preprocess import resize_symbol

input_images_folder = "captcha images"
output_folder = "single symbols"

input_images = glob.glob(os.path.join(input_images_folder, "*"))
counts = {}

for (i, input_image) in enumerate(input_images):
    print("[INFO] processing image {}/{}".format(i + 1, len(input_images)))
    
    image_name = os.path.basename(input_image)
    captcha_text = image_name.split(".")[0]    
    
    image = cv2.imread(input_image)    
    
    image = color_to_black(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value = 255)
    height, width = gray.shape[:2]  
    gray = clear_background(gray)
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour_regions = split_into_pieces(contours, width, height)
    
    contour_regions = sorted(contour_regions, key=lambda x: x[0])
    for symbol_box, symbol_text in zip(contour_regions, captcha_text):
        x, y, w, h = symbol_box
        symbol_image = gray[y: y + h, x: x + w]
        symbol_image = resize_symbol(symbol_image, 28)
        save_path = os.path.join(output_folder, symbol_text)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        count = counts.get(symbol_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, symbol_image)
        counts[symbol_text] = count + 1
        
        
    
    
    
    