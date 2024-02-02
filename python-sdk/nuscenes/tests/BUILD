py_test(
    name = "assert_download",
    srcs = ["assert_download.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//python-sdk/nuscenes",
    ],
    imports = ["../.."], # add this to PYTHONPATH, relative to this BUILD file
)

py_test(
    name = "test_lidarseg",
    srcs = ["test_lidarseg.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//python-sdk/nuscenes",
    ],
    imports = ["../.."], # add this to PYTHONPATH, relative to this BUILD file
    env = {
        "NUSCENES": "/data/sets/nuscenes",
    },
)

py_test(
    name = "test_nuscenes",
    srcs = ["test_nuscenes.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//python-sdk/nuscenes",
    ],
    imports = ["../.."], # add this to PYTHONPATH, relative to this BUILD file
    env = {
        "NUSCENES": "/data/sets/nuscenes",
    },
)

py_test(
    name = "test_predict_helper",
    srcs = ["test_predict_helper.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//python-sdk/nuscenes",
        # TODO add nuscenes.prediction
    ],
    imports = ["../.."], # add this to PYTHONPATH, relative to this BUILD file
    env = {
        "NUSCENES": "/data/sets/nuscenes",
    },
)
