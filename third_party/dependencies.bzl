load("@//third_party/googletest:googletest.bzl", "googletest")
load("@//third_party/tensorflow:tensorflow.bzl", "tensorflow")
load("@//third_party/zlib:zlib.bzl", "zlib")

def third_party_dependencies():
    """ Load 3rd Party Dependencies """

    googletest()
    tensorflow()
    zlib()
