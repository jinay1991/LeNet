#!/usr/bin/python3

import os

import tensorflow as tf


def LeNet():
    """
    Constructs Keras Model Object
    """
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5),
                                     activation='relu', input_shape=(32, 32, 3), name="C1"))
    model.add(tf.keras.layers.MaxPooling2D(name="S2"))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', name="C3"))
    model.add(tf.keras.layers.MaxPooling2D(name="S4"))
    model.add(tf.keras.layers.Flatten(name="C5_1"))
    model.add(tf.keras.layers.Dense(units=120, activation='relu', name="C5_2"))
    model.add(tf.keras.layers.Dense(units=84, activation='relu', name="F6"))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax', name="Output"))

    return model


if __name__ == "__main__":
    # Prepare Dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Create the convolution network
    model = LeNet()

    model.summary()

    # Compile and train the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))

    # Evalaute the model
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print("test accuracy: ", test_acc)

    # Save model
    if not os.path.exists("saved_model"):
        os.mkdir("saved_model")
    model.save("saved_model")
