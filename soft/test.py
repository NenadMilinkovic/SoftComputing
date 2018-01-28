import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import os
import cv2

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import Adagrad


def create_ann():
    '''Implementacija veštačke neuronske mreže sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential ()
    ann.add (Dense (512, input_dim=2352, activation='sigmoid'))
    ann.add (Dense (52, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train):
    '''Obucavanje vestacke neuronske mreze'''
    X_train = np.array (X_train, np.float32)  # dati ulazi
    y_train = np.array (y_train, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD (lr=0.01, momentum=0.9)
    ann.compile (loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit (X_train, y_train, epochs=2000, batch_size=1, verbose=0, shuffle=False)

    return ann

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255.
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann

def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)


def select_region(image_bin):
    x = 1130;
    y = 1280;
    h = 340;
    w = 200;
    region = image_bin[y:y + h + 1, x:x + w + 1]
    region=resize_region(region)
    region = scale_to_range(region)
    region=matrix_to_vector(region)
    return  region

def select_test_region(image_bin):
    x = 1812;
    y = 111;
    h = 125;
    w = 249;
    region = image_bin[y:y + h + 1, x:x + w + 1]
    region=resize_region(region)
    region=scale_to_range(region)
    region=matrix_to_vector(region)

    return  region





def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros ((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum (axis=1)
    rect[0] = pts[np.argmin (s)]
    rect[2] = pts[np.argmax (s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff (pts, axis=1)
    rect[1] = pts[np.argmin (diff)]
    rect[3] = pts[np.argmax (diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points (pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt (((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt (((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max (int (widthA), int (widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt (((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt (((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max (int (heightA), int (heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array ([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform (rect, dst)
    warped = cv2.warpPerspective (image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array=[]
    box = []
    pts = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect (contour)
        rect,f,angle=cv2.minAreaRect(contour)
        cv2.rectangle (image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

        size=cv2.contourArea(contour)
        peri=cv2.arcLength(contour,True)
        approx=cv2.approxPolyDP(contour,0.1*peri,True)

        if(size>4500):
           #print (size)
           #print (len (approx))
           #print ('-------')
           list=[]

           for idx,i in enumerate(approx):
               list.append(approx[idx][0])
           pts=np.array(list,np.float32)

           regions_array.append(image_bin[y:y+h+1,x:x+w+1])
           print(pts)


    #regions_array = sorted (regions_array, key=lambda item: item[1][0])
   # sorted_regions = sorted_regions = [region[0] for region in regions_array]

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, regions_array,pts,angle


def matrix_to_vector(image):
    return image.flatten()


def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    return image/255


def convert_output(names):
    nn_outputs = []
    for index in range(len(names)):
        output = np.zeros(len(names))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def winner(output): # output je vektor sa izlaza neuronske mreze

    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result


def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor (image, cv2.COLOR_RGB2GRAY)
    #blur = cv2.GaussianBlur (gray, (5, 5), 0)

    img_w, img_h = np.shape (image)[:2]
    bkg_level = gray[int (img_h / 100)][int (img_w / 2)]
    thresh_level = bkg_level + 60

    retval, thresh = cv2.threshold (gray, thresh_level, 255, cv2.THRESH_BINARY)


    return thresh

def select_test_region(contour):
    x = 0;
    y = 0;
    h = 600;
    w = 300;
    region = contour[y:y + h + 1, x:x + w + 1]
    plt.imshow(region,'gray')
    plt.show()
    region = resize_region (region)
    region = scale_to_range (region)
    region = matrix_to_vector (region)
    return region

def select_sign(contour):
    x = 0;
    y = 0;
    h = 600;
    w = 300;
    region = contour[y:y + h + 1, x:x + w + 1]
    #plt.imshow(region)
    #plt.show()
    region = resize_region (region)
    region = scale_to_range (region)
    region = matrix_to_vector (region)
    return region

def testiranje_rotacije():
    pictures=os.listdir('images')
    data = []
    labels = []
    for card in pictures:
        cardname = card.split ('.')[0]
        labels.append (cardname)
        image_color = load_image ('images/' + card)
        image = preprocess_image (image_color)
        img, contours, box, angle = select_roi (image_color.copy (), image)

        warp=four_point_transform(img,(box))
        region = select_sign (warp)

        data.append (matrix_to_vector (region))

    te_labels = convert_output (labels)
    ann = create_ann ()
    print ("Mreza kreirana")

    ann = train_ann (ann, data, te_labels)
    ann.save ('NeuralNetwork.h5')

    ann=load_model('NeuralNetwork.h5')
    for i in pictures:
        image_color = load_image ('images/' + i)
        image = preprocess_image (image_color)
        img, contours, box, angle = select_roi (image_color.copy (), image)

        warp = four_point_transform (img, (box))
        region =[select_test_region (warp)]

        result = ann.predict (np.array (region))
        print (result)
        print (display_result (result, labels))
testiranje_rotacije()

###secenje uglova
#peri=cv2.arcLength(contours[0],True)
#approx=cv2.approxPolyDP(contours[0],0.01*peri,True)
#pts=np.float32(approx)
#print(approx)
#average=np.sum(pts,axis=0)/len(pts)
#cent_x = int(average[0][0])
#cent_y = int(average[0][1])

#x,y,w,h = cv2.boundingRect(contours[0])
#wrap=flattener(contours[0], pts, w, h)

#corner=wrap[0:84,0:32]
#plt.imshow(corner)
#plt.show()