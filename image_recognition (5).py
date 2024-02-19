import cv2
from matplotlib import pyplot as plt


def image_read(FileIm):
    image = cv2.imread(FileIm)
    plt.imshow(image)
    plt.show()
    return image


def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_normalized = cv2.normalize(gray, None, 0, 150, cv2.NORM_MINMAX)  # прибрався напис на першому зображенні
    gray = cv2.GaussianBlur(gray_normalized, (3, 3), 0)
    edged = cv2.Canny(gray, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    plt.imshow(closed)
    plt.show()
    return closed


def image_contours(image_entrance):
    cnts = cv2.findContours(image_entrance.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    return cnts


def is_contour_round(c, epsilon=0.01):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon * peri, True)
    return len(approx) > 7  # Check if the contour is round


def image_recognition(image_entrance, image_cont, file_name):
    total = 0
    for c in image_cont:
        if is_contour_round(c):
            cv2.drawContours(image_entrance, [c], -1, (0, 255, 0), 2)
            total += 1

    print("Знайдено {0} сегмент(а) круглих об'єктів".format(total))
    cv2.imwrite(file_name, image_entrance)
    plt.imshow(image_entrance)
    plt.show()


if __name__ == '__main__':
    image_entrance = image_read("Coins/image_1.jpg")
    image_exit = image_processing(image_entrance)
    image_cont = image_contours(image_exit)
    image_recognition(image_entrance, image_cont, "Coins/image_recognition_1.jpg")

    image_entrance = image_read("Coins/image_2.jpg")
    image_exit = image_processing(image_entrance)
    image_cont = image_contours(image_exit)
    image_recognition(image_entrance, image_cont, "Coins/image_recognition_2.jpg")
