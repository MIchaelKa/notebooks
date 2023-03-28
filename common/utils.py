import cv2


def resize_shortest(image, size, interpolation):
    src_height, src_width, _ = image.shape

    if src_height > src_width:
        new_height = src_height * size / src_width
        new_size = (int(size), int(new_height))
    else:
        new_width = src_width * size / src_height
        new_size = (int(new_width), int(size))
    
    ratio = src_width / new_size[0]
    
    resized_image = cv2.resize(image, new_size, interpolation=interpolation)
    
    return ratio, resized_image