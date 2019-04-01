import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.preprocessing.image import array_to_img
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,BatchNormalization,concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


from keras import backend as K
K.clear_session()
from processor import dataProcess


class mynet(object):

	def __init__(self, img_rows = 512, img_cols = 512):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_label = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_label, imgs_test

	def create_net(self):

		inputs = Input((self.img_rows, self.img_cols,1))
		print("inputs shape:",inputs.shape)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		print ("conv1 shape:",conv1.shape)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		print ("conv1 shape:",conv1.shape)
		batchnor1=BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-3,center=True,scale=True,beta_initializer='zeros',gamma_initializer='zeros',moving_mean_initializer='zeros',moving_variance_initializer='ones')(conv1)
		print("batchnor1 shape:", batchnor1.shape)
		pool1 = MaxPooling2D(pool_size=(2, 2))(batchnor1)
		print ("pool1 shape:",pool1.shape)

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		print ("conv2 shape:",conv2.shape)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		print ("conv2 shape:",conv2.shape)
		batchnor2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True,
									   beta_initializer='zeros', gamma_initializer='zeros',
									   moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv2)
		print("batchnor2 shape:", batchnor2.shape)
		pool2 = MaxPooling2D(pool_size=(2, 2))(batchnor2)
		print ("pool2 shape:",pool2.shape)

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		print ("conv3 shape:",conv3.shape)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		print ("conv3 shape:",conv3.shape)
		batchnor3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True,
									   beta_initializer='zeros', gamma_initializer='zeros',
									   moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv3)
		print("batchnor3 shape:", batchnor3.shape)
		pool3 = MaxPooling2D(pool_size=(2, 2))(batchnor3)
		print("pool3 shape:",pool3.shape)

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		batchnor4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True,
									   beta_initializer='zeros', gamma_initializer='zeros',
									   moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv4)
		print("batchnor4 shape:", batchnor4.shape)
		drop4 = Dropout(0.5)(batchnor4)
		print("drop4 shape:", drop4.shape)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
		print("pool4 shape:", pool4.shape)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		batchnor5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True,
									   beta_initializer='zeros', gamma_initializer='zeros',
									   moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv5)
		print("batchnor5 shape:", batchnor5.shape)
		drop5 = Dropout(0.5)(batchnor5)
		print("drop5 shape:", drop5.shape)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(batchnor5))
		print("up6 shape:", up6.shape)
		concat6 = concatenate([conv4,up6], axis = 3)
		print("merge6 shape:", concat6.shape)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concat6)
		print("conv6 shape:", conv6.shape)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
		print("conv6 shape:", conv6.shape)
		batchnor6= BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True,
									   beta_initializer='zeros', gamma_initializer='zeros',
									   moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv6)
		print("batchnor6 shape:", batchnor6.shape)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(batchnor6))
		print("up7 shape:", up7.shape)
		concat7 = concatenate([conv3,up7],axis = 3)
		print("merge7 shape:",concat7.shape)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concat7)
		print("conv7 shape:", conv7.shape)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
		print("conv7 shape:", conv7.shape)
		batchnor7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True,
									   beta_initializer='zeros', gamma_initializer='zeros',
									   moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv7)
		print("batchnor7 shape:", batchnor7.shape)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(batchnor7))
		print("up8 shape:", up8.shape)
		concat8 = concatenate([conv2,up8],axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concat8)
		print("conv8 shape:", conv8.shape)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
		print("conv8 shape:", conv8.shape)
		batchnor8 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True,
									   beta_initializer='zeros', gamma_initializer='zeros',
									   moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv8)
		print("batchnor8 shape:", batchnor8.shape)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(batchnor8))
		print("up9 shape:", up9.shape)
		concat9 = concatenate([conv1,up9], axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concat9)
		print("conv9 shape:", concat9.shape)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		print("conv9 shape:", conv9.shape)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		print("conv9 shape:", conv9.shape)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
		print("conv10 shape:", conv10.shape)
		model = Model(input = inputs, output = conv10)

		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

		return model


	def train(self):

		print("Loading data......")
		imgs_train, imgs_label, imgs_test = self.load_data()
		print("Load data done!")
		model = self.create_net()

		model_checkpoint = ModelCheckpoint('new_net.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, imgs_label, batch_size=4, nb_epoch=1000, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

		print('Predicting test data.....')
		imgs_test_predict = model.predict(imgs_test, batch_size=1, verbose=1)
		np.save('../test_predict/imgs_test_predict.npy', imgs_test_predict)

	def save_img(self):

		print("Save the test result.")
		imgs = np.load('../test_predict/imgs_test_predict.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save("../test_predict/%d.jpg"%(i))

if __name__ == '__main__':
	pixelunet = mynet()
	pixelunet.train()
	pixelunet.save_img()








