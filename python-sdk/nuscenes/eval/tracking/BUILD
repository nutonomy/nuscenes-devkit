py_binary(
    name = "evaluate",
    srcs = [
        "evaluate.py",
        # Even though the source files below should already be part of the nuscenes/eval py_library dep,
        # we need to add them to the sources here, too. It seems like bazel prevents them from
        # being part of the globbing there because they are in the same directory as the source
        # of this depending py_binary here.
        "algo.py",
        "constants.py",
        "data_classes.py",
        "loaders.py",
        "metrics.py",
        "mot.py",
        "render.py",
        "utils.py",
    ],
    deps = ["//python-sdk/nuscenes/eval", "//python-sdk/nuscenes"],
    visibility = ["//visibility:public"],
    data = [":configs/tracking_nips_2019.json"],
    imports = ["../../.."], # add this to PYTHONPATH, relative to this BUILD file
)
