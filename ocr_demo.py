import cv2
from PIL import Image
import pytesseract
import argparse


def parse_lines(text):
    """
    Function to convert raw text from tesseract into document structure format - Block, Paragraphs, lines, words
    Args:
        text: raw text data from tesseract

    Returns:
        Structured text data
    """
    data = {}
    for i in range(len(text['line_num'])):
        if not (text['text'][i] == '' or text['text'][i].isspace()):
            if text['block_num'][i] in data:

                if text['line_num'][i] in data[text['block_num'][i]]:
                    data[text['block_num'][i]][text['line_num'][i]].append((text['text'][i], text['left'][i], text['top'][i], text['width'][i], text['height'][i]))
                else:
                    # lastLineKey = text['line_num'][i]
                    # line[text['line_num'][i]] = []
                    data[text['block_num'][i]][text['line_num'][i]] = [(text['text'][i], text['left'][i], text['top'][i], text['width'][i], text['height'][i])]
                    # line[lastLineKey].append()

            else:
                data[text['block_num'][i]] = {}
                data[text['block_num'][i]][text['line_num'][i]] = [(text['text'][i], text['left'][i], text['top'][i], text['width'][i], text['height'][i])]

    linedata = {}
    idx = 0
    for _, b  in data.items():
        for _, l in b.items():
            linedata[idx] = l
            idx += 1

    return linedata

def ocr_image(filePath):
    """
    Function to read text from image using Tesseract.

    Args:
        filePath: path to image file

    Returns:
        OCRed Text data
    """
    print("Main function invoked!")
    img = cv2.imread(filePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_pil = Image.fromarray(gray)

    # OCR image
    config = ('-l eng --oem 1')
    text = pytesseract.image_to_data(gray_pil, config=config, output_type='dict')
    lines = parse_lines(text)

    return lines

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-image", "--image", help="Input Image Path")
    args = parser.parse_args()
    filePath = args.image
    text_lines = ocr_image(filePath)

    print("Print text lines recognised by Tesseract\n")
    for key, arr in text_lines.items():
        l = ''
        for item in arr:
            l += item[0] + " "
        print(l)