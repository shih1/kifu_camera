import cv2
import sys
import numpy as np
TMP_IMAGE_PATH = "../images/IMG-3660.jpg"

#percent by which the image is resized
SCALE_PERCENT = 40

def main():

    # Read image
    image = cv2.imread(TMP_IMAGE_PATH)

    # Debugging windwos 
    cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
    cv2.namedWindow('tmpImg', cv2.WINDOW_NORMAL)
    cv2.namedWindow('tmpImg1', cv2.WINDOW_NORMAL)

    # Setup Kernels for Smoothing and Dilation
    kernel_gaussian = np.ones((5,5),np.float32)/25
    kernel_dilation = np.ones((5,5),np.float32)
    
    # Scale Parameters 
    width = int(image.shape[1] * SCALE_PERCENT / 100)
    height = int(image.shape[0] * SCALE_PERCENT / 100)
    dsize = (width, height)

    # Image Resizing
    image = cv2.resize(image, dsize)
    og_image =image
    
    image = cv2.filter2D(image,-1,kernel_gaussian)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # openCV Enum for grayscale images
    
    # Canny Edge Detection
    '''
    minVal = 50 - empircally chosen
    maxVal = 150 - empircally chosen
    apetureSize = 3

    Apeture size (range[3,7]) is chosen to be the smallest possible. 
    Increasing apeture size will **increase** the number of edges found. 

    It is recommended to change the smoothing algorithm _before_ adjusting these parameters.
    '''
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    cv2.imshow("tmpImg1", edges) 

    # Morphological Transformations
    '''
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

    Image dilation is used to increase the size of the existing lines. The kernel selected will dilate _white_ pixels. 

    Increasing iterations will further dilate the edge image. 
    '''
    dilation = cv2.dilate(edges,kernel_dilation,iterations = 1)
    cv2.imshow("tmpImg", dilation) 
    

    '''
    TODO(Chris): Further morphological transformations, specifically Closing and Opening (see link above), will further increase the clarity of the edge image. It may be possible to reduce the intensity of the dilation algorithm after Closing andn Opening are added. 
    '''

    # Probablistic Hough Transform
    '''
    Threshold, minLineLength, and maxLineGap are tuned to detect lines in the edge image. 

    threshold is the minimum # of votes needed for a line to be considered a line 
    minLineLength is self explanantory
    minLineGap is the biggest allowoed gap between two pixels of the same line
    '''
    threshold = 120
    minLineLength = 200
    maxLineGap = 30
    # Note: Lines is a 3D numpy ndarray
    lines = cv2.HoughLinesP(dilation,1,np.pi/(180),threshold,minLineLength=minLineLength, maxLineGap=maxLineGap)

    for i in range(lines.shape[0]):
        x1 = int(lines[i, :, 0])
        y1 = int(lines[i, :, 1])
        x2 = int(lines[i, :, 2])
        y2 = int(lines[i, :, 3])
        cv2.line(og_image,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("finalImg", og_image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()