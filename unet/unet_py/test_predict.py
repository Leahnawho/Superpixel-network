'''
from unet import *
from data import *

mydata = dataProcess(512,512)

imgs_test = mydata.load_test_data()

myunet = myUnet()

model = myunet.get_unet()

model.load_weights('unet.hdf5')

imgs_mask_test = model.predict(imgs_test, verbose=1)

np.save('imgs_mask_test.npy', imgs_mask_test)
'''

from New_net import *
from processor import *
import numpy as np

if __name__ == "__main__":
    mydata = dataProcess(512,512)
    imgs_test = mydata.load_test_data()
    pixelunet = mynet()
    model = pixelunet.create_net()
    model.load_weights('new_net.hdf5')
    imgs_test_predict = model.predict(imgs_test, verbose=1)
    np.save('../test_predict/imgs_test_predict.npy', imgs_test_predict)
    pixelunet.save_img()