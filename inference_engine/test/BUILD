load("@rules_cc//cc:defs.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "unit_test",
    srcs = glob(
        ["*.cpp"],
    ),
    copts = [
        "-std=c++14",
        "-Wall",
        "-Werror",
    ],
    data = [
        "@audio_example//file",
    ],
    linkstatic = True,
    deps = [
        "//inference_engine/argument_parser",
        "//inference_engine/audio",
        "//inference_engine/inference_engine",
        "//inference_engine/logging",
        "@audionamix//:wave",
        "@googletest//:gtest_main",
    ],
)
