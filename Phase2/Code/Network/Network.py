from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Activation, Dropout, InputLayer, Conv2D, MaxPooling2D


def get_homography_model():
    conv_size = 3
    pool_size = 2
    no_filters = [64, 128]
    input_shape = (128, 128, 2)
    model = Sequential()
    # Convolution layer 1
    model.add(InputLayer(input_shape))
    model.add(Conv2D(filters=no_filters[0],
              kernel_size=conv_size, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Convolution layer 2
    model.add(Conv2D(filters=no_filters[0],
              kernel_size=conv_size, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Pool layer 1
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    # Convolution layer 3
    model.add(Conv2D(filters=no_filters[0],
              kernel_size=conv_size, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Convolution layer 4
    model.add(Conv2D(filters=no_filters[0],
              kernel_size=conv_size, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Pool layer 2
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    # Convolution layer 5
    model.add(Conv2D(filters=no_filters[1],
              kernel_size=conv_size, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Convolution layer 6
    model.add(Conv2D(filters=no_filters[1],
              kernel_size=conv_size, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Pool layer 3
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    # Convolution layer 7
    model.add(Conv2D(filters=no_filters[1],
              kernel_size=conv_size, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Convolution layer 8
    model.add(Conv2D(filters=no_filters[1],
              kernel_size=conv_size, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # FC layer 1
    model.add(Flatten())
    model.add(Dense(units=1024))
    model.add(Activation(activation='relu'))
    model.add(BatchNormalization())
    # Drop out layer
    model.add(Dropout(rate=0.5))
    # FC layer 2
    model.add(Dense(units=8, activation='relu'))
    model.compile(optimizer='adam', loss='mse', metrics=["mae"])
    return model
