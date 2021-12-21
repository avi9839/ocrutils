## OCR-Utils

With growing need of digitalization, many businesses are in document management domain, are looking for new ways to digitalize the documents data by uploading them to their databases and making it accessible to their users from different devices in real time. In order to make the document quickly searchable, its text contents needs to parsed using OCR tools.
Most of these documents are created and uploaded by users with different devices and all the documents captured from mobile device cameras contain different issues like Low illumination, non-uniform light, Shadows, Glares and such issues affect the performance of OCR tools.

This repository provides document image processing and quality enhancement methods for OCR tools.

#### Requirements:
1. OpenCV
2. scikit-learn
3. scikit-image
4. pytesseract
5. numpy

### Examples and Uses:

### Image Binarization Demo
#### Binarization algorithm demo
```` python
    filePath = 'img/ocr_image.jpg'
    img = cv2.imread(filePath)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Instantiate the Image Binarization Object
    image_binarization = ImageBinarization()

    bw_img = image_binarization.otsuThreshold(gray_img)
    cv2.imshow("OTSU BW", bw_img)
    cv2.waitKey(0)

    bw_img = image_binarization.adaptive(gray_img)
    cv2.imshow("Adaptive BW", bw_img)
    cv2.waitKey(0)

    bw_img = image_binarization.bradley_roth_numpy(gray_img)
    cv2.imshow("Bradley BW", bw_img)
    cv2.waitKey(0)

    bw_img = image_binarization.niblack(gray_img)
    cv2.imshow("Niblack BW", bw_img)
    cv2.waitKey(0)

    bw_img = image_binarization.sauvola(gray_img)
    cv2.imshow("Sauvola BW", bw_img)
    cv2.waitKey(0)
````
Binarization results:
![Alt text](img/binarization_results.jpg?raw=true "Binarization Results")
As shown in above figure, we can try with different binarization algorithms with different hyperparameters to improve the document quality best suitable for OCR.


#### Lines/Table detection and removal:
```python
    filePath = '../img/table_img.png'
    remove_noise = RemoveNoise()
    img = cv2.imread(filePath)
    res_img = remove_noise.removeNoise(img)
    cv2.imshow("Image Without Table", res_img)
    cv2.waitKey(0)
```
Original Image
![Alt text](img/table_img.png?raw=true "Table")

Image without any line/Table
![Alt text](img/Image_Without_Table.png?raw=true "Table Removed")

#### OCR Demo:
![Alt text](img/ocr_image.jpg?raw=true "OCR Image")

```commandline
python --image img/ocr_image.jpg
```
OCR Output:
```text
When in 1969 Bowers was awarded the Gold Medal of the Biblio- 
graphical Society in London, John Carter's citation referred to the 
Principles as “majestic,” called Bowers's current projects “formidable,” 
said that he had “imposed critical discipline” on the texts of several 
authors, described Studies in Bibliography as a “great and continuing 
achievement,” and included among his characteristics “uncompromising 
seriousness of purpose” and “professional intensity.” Bowers was not 
unaccustomed to such encomia, but he had also experienced his share of 
attacks: his scholarly positions were not universally popular, and he 
expressed them with an aggressiveness that almost seemed calculated to
```
