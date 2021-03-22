import cv2
import sys
import numpy as np


class BoardSegmentation(object):
    ''' 
    **Example Usage**

    bs = BoardSegmentation(resizePercentage = 50, debug = True)
    bs.load_image()
    bs.find_edges()
    lines = bs.getLines()

    '''
    def __init__(self, resizePercentage = 40, debug = False):
        self.clear()

        # Parameters
        self.scale_percent = resizePercentage
        self.debug = debug
        #Kernels
        self.kernel_gaussian = np.ones((5,5),np.float32)/25
        self.kernel_dilation = np.ones((5,5),np.float32)

        
        if self.debug: 
            cv.namedWindow("OriginalImage")
            cv.namedWindow("ResizedImage")
            cv.namedWindow("SmoothedImage")
            cv.namedWindow("GrayscaleImage")
            cv.namedWindow("EdgeImage")
            cv.namedWindow("DilatedImage")
            cv.namedWindow("LineImage")

    def clear(self):
        self.image = None
        self.resized_image = None

        self.smoothed_image = None
        self.grayscale_image = None
        self.edge_image = None

        self.dilated_image = None
        self.line_image = None

    def load_image(self, imagePath):
        # poplate self.image and self.resized image
        self.clear()

        self.image  = cv.imread(imagePath)
        width = int(self.image.shape[1] * self.scale_percent / 100)
        height = int(self.image.shape[0] * self.scale_percent / 100)
        
        dsize = (width, height)
        self.resized_image = cv2.resize(image, dsize)

        if self.debug: 
            cv2.imshow("OriginalImage", self.image) 
            cv2.imshow("ResizedImage", self.resized_image)
            cv2.waitKey(0)

    def find_edges(self):
        '''
        Smooth and Detect edges

        **Canny Parameters Rationale**
        minVal = 50 - empircally chosen
        maxVal = 150 - empircally chosen
        apetureSize = 3

        Apeture size (range[3,7]) is chosen to be the smallest possible. 
        Increasing apeture size will **increase** the number of edges found. 
        It is recommended to change the smoothing algorithm _before_ adjusting these parameters.
        '''
        self.smoothed_image = cv2.filter2D(image,-1,self.kernel_gaussian)
        self.grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.edge_image = cv2.Canny(gray, 50, 150, apertureSize = 3)
        
        if self.debug: 
            cv2.imshow("SmoothedImage", self.smoothed_image) 
            cv2.imshow("GrayscaleImage", self.grayscale_image)
            cv2.imshow("EdgeImage", self.edge_image)
            cv2.waitKey(0)

    def get_lines(self):
        '''
        Morph and find lines

        Threshold, minLineLength, and maxLineGap are tuned to detect lines in the edge image. 
        threshold is the minimum # of votes needed for a line to be considered a line 
        minLineLength is self explanantory
        minLineGap is the biggest allowed gap between two pixels of the same line
        '''

        if not self.edge_image:
            sys.exit("Must find edges first! Try running self.find_edges()")

        threshold = 120
        minLineLength = 200
        maxLineGap = 30
        
        # Lines is a 3D numpy ndarray
        lines = cv2.HoughLinesP(dilation,1,np.pi/(180),threshold,minLineLength=minLineLength, maxLineGap=maxLineGap)
        self.dilated_image = cv2.dilate(self.edge_image, self.kernel_dilation, iterations = 1)

        if self.debug:
            cv2.imshow("DilatedImage", self.dilated_image) 
        
            # Draw Lines on original image
            self.line_image = self.resized_image
            for i in range(lines.shape[0]):
                x1 = int(lines[i, :, 0])
                y1 = int(lines[i, :, 1])
                x2 = int(lines[i, :, 2])
                y2 = int(lines[i, :, 3])
                cv2.line(self.line_image, (x1,y1),(x2,y2),(0,255,0), 2)
            cv2.imshow("LineImage", self.line_image)
            cv2.waitKey(0)

        return lines
