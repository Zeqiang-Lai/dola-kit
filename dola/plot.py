import cv2


def add_text(image, text):
    # overlay space
    x, y, w, h = 0, 0, 100, 50

    # alpha, the 4th channel of the image
    alpha = 0.5

    overlay = image.copy()
    output = image.copy()

    # corner
    cv2.rectangle(overlay, (x, x), (x + w, y + h), (0, 0, 0), -1)

    # putText
    cv2.putText(overlay, text, (x + int(w / 10), y + int(h)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # apply the overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output
