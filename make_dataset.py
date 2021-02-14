import glob
import os.path

import cv2

from preprocess import make_counters, resize_symbol

input_images_folder = "captcha images"
output_folder = "single symbols"

input_images = glob.glob(os.path.join(input_images_folder, "*"))
counts = {}

for (i, input_image) in enumerate(input_images):
    print("[INFO] processing image {}/{}".format(i + 1, len(input_images)))
    
    image_name = os.path.basename(input_image)
    captcha_text = image_name.split(".")[0]    
    
    image = cv2.imread(input_image)

    contour_regions, gray = make_counters(image)
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
