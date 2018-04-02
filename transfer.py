from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input, AveragePooling2D
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.1,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (299, 299),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (299, 299),
                                            batch_size = 32,
                                            class_mode = 'binary')

input_tensor = Input(shape=(299, 299, 3))

network = InceptionV3(include_top=False, weights="imagenet", input_tensor=input_tensor)

for layer in network.layers:
    layer.trainable = False

my_network = network.output
my_network = AveragePooling2D((4, 4), border_mode='valid', name='avg_pool')(my_network)
my_network = Dropout(0.3)(my_network)
my_network = Flatten()(my_network)
my_network = Dense(1, activation="sigmoid")(my_network)

model = Model(inputs=network.input, outputs=my_network)
model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

model.fit_generator(training_set,
                         samples_per_epoch=8000,
                         nb_epoch=15,
                         validation_data=test_set,
                         nb_val_samples=2000)
