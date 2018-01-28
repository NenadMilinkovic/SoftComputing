import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import os
import cv2

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model


def create_ann():
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(128,activation='sigmoid'))
    ann.add(Dense(52, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train):
    '''Obucavanje vestacke neuronske mreze'''
    X_train = np.array(X_train, np.float32)  # dati ulazi
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    print ("Pocelo obucavanje")
    # obucavanje neuronske mreze
    training=ann.fit(X_train, y_train, epochs=2000, batch_size=1, verbose=0, shuffle=False)
    print(training.history)
    print("Zavrseno")
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


def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    x=1130;
    y=1280;
    h=340;
    w=200;
    region = image_bin[y:y + h + 1, x:x + w + 1]
    region=scale_to_range(region)
    region=matrix_to_vector(region)
    regions_array.append([resize_region(region), (x, y, w, h)])
    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions


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
image_color1 = load_image('test/test.jpg')
plt.imshow(image_color1)
plt.show()

pictures=os.listdir('images')
data = []
labels = []
for card in pictures:
    cardname=card.split('.')[0]
    labels.append(cardname)
    image_color = load_image('images/'+card)
    img = invert(image_bin(image_gray(image_color)))
    img_bin = erode(dilate(img))
    numbers = select_region(img)
    testImage=[matrix_to_vector(numbers)]
    data.append(matrix_to_vector(numbers))
print("Pokupio regione")
te_labels=convert_output(labels)

print(len(data))
print(len(te_labels))

ann=create_ann()
print("Mreza kreirana")

print("Mreza kreirana")
print(len(data))
print("Ulazi")

ann=train_ann(ann,data,te_labels)

ann.save('NeuralNetwork.h5')
img2 = invert(image_bin(image_gray(image_color1)))
img_bin1 = erode(dilate(img2))
region =[select_test_region(img2)]

result=ann.predict(np.array(region))
print( result)
print( display_result(result,labels))
#im = cv2.imread('images/poker.jpg')
#gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#blur = cv2.GaussianBlur(gray, (1, 1), 1000)
#flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

#image,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#image=im.copy()
#cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
#plt.imshow(image)
#plt.show()
#contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

#approx=0;
#for i in range(len(contours)):
 #   card = contours[i]
  #  peri = cv2.arcLength(card,True)
   # approx = cv2.approxPolyDP(card,0.02*peri,True)
    #rect = cv2.minAreaRect(contours[2])
    #r = cv2.cv.boxPoints(rect)

#h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
#transform = cv2.getPerspectiveTransform(approx,h)
#warp = cv2.cv.warpPerspective(im,transform,(450,450))
#plt.imshow(warp)
#plt.show()
