import cv2
import imutils


def color_to_black(image):
    height, width = image.shape[:2]    
    for row in range(0, height, 1):
        for col in range(0, width, 1):
            if image[row, col][0] < 129 or image[row, col][1] < 129 or image[row, col][2] < 129:
                image[row, col] = [0, 0, 0]         
            else:
                image[row, col] = [255, 255, 255]
    return image


def clear_background(gray):
    height, width = gray.shape[:2]    
    for row in range(0, height - 1, 1):
        for col in range(0, width - 1, 1):
            if gray[row, col] == 0: 
                if gray[row + 1, col + 1] == 255:
                    gray[row, col] = 255
                if gray[row, col + 1] == 255:
                    gray[row, col] = 255
                if gray[row + 1, col] == 255:
                    gray[row, col] = 255

    for row in range(height - 1, -1, -1):
        for col in range(width - 1, 0, -1):
            if gray[row, col] == 0: 
                if gray[row - 1, col - 1] == 255:
                    gray[row, col] = 255
                if gray[row, col - 2] == 255:
                    gray[row, col] = 255
                if gray[row - 1, col] == 255:
                    gray[row, col] = 255
    return gray


def split_into_pieces(contours, width, height):
    regions = []    
    item = (180, 100, 0, 0)
    max_x = 0
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if h < 30 and w < 20:
            continue

        min_x = min(item[0], x)
        max_x = max(max_x, x + w)
        min_y = min(item[1], y)
        max_y = max(item[3], h)
        regions.append((min_x, min_y, max_x - min_x, max_y))
        if max_x - min_x > width // 2 and max_y > height // 3:
            w = (max_x - min_x) // 4
            h = min_y + max_y
            regions = [
                (min_x, min_y, w, h),
                (min_x + w, min_y, w, h),
                (max_x - 2 * w, min_y, w, h),
                (max_x - w, min_y, w, h),
            ]
    return regions


def resize_symbol(symbol_image, size):
    symbol_image = imutils.resize(symbol_image, height = size)
    w = int((size - symbol_image.shape[0]) / 2.0)
    h = int((size - symbol_image.shape[1]) / 2.0)
    symbol_image = cv2.copyMakeBorder(symbol_image, h, h, w, w, cv2.BORDER_CONSTANT, value=255)
    symbol_image = cv2.resize(symbol_image, (size, size))
    return symbol_image


def make_counters(image):
    image = color_to_black(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    height, width = gray.shape[:2]
    gray = clear_background(gray)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour_regions = split_into_pieces(contours, width, height)

    contour_regions = sorted(contour_regions, key=lambda x: x[0])
    return contour_regions, gray
