import cv2
import imutils
    
def color_to_black(image):
    height, width = image.shape[:2]    
    for row in range(0, height, 1):
        for col in range(0, width, 1):
            if image[row, col][0] < 129  or image[row, col][1]  < 129 or image[row, col][2] < 129:
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
    maxX = 0
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)                
        if h > 30 or w > 20:
            minX = min(item[0], x)
            maxX = max(maxX, x + w)
            minY = min(item[1], y)
            maxH = max(item[3], h)
            item = (minX, minY, 0, maxH)
    regions.append((minX, minY, maxX - minX , maxH))
    if maxX - minX > width // 2 and maxH > height // 3:
        W = (maxX - minX) // 4
        H = minY + maxH
        regions = [(minX, minY, W, H),(minX + W, minY, W, H),(maxX - 2 * W, minY, W, H),(maxX - W, minY, W, H)]
    return regions
    
    
def resize_symbol(simbol_image, size):
    simbol_image = imutils.resize(simbol_image, height = size)
    W = int((size - simbol_image.shape[0]) / 2.0)
    H = int((size - simbol_image.shape[1]) / 2.0)
    simbol_image = cv2.copyMakeBorder(simbol_image, H, H, W, W, cv2.BORDER_CONSTANT, value = 255)
    simbol_image = cv2.resize(simbol_image, (size, size))
    return simbol_image 
    
    
    
    
    
    