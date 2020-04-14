#!/usr/bin/python3
import tensorflow as tf

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--saved_model_dir", help="SavedModel directory", required=True)
    parser.add_argument("-t", "--tflite_model", help="TFLiteModel directory", default="deployed_model.tflite")
    args = parser.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)

    with open(args.tflite_model, "wb") as fp:
        fp.write(converter.convert())
