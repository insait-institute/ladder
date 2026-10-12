load("//build_support/python:pip_defs.bzl", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "video_utils",
    srcs = glob(
        ["**/*.py"],
        exclude = ["**/test/**"],
    ),
    pip_deps = [
        "av",
        "numpy",
        "pillow",
        "tqdm",
    ],
    deps = [
        "//ladder/core/common:np_utils",
        "//ladder/core/common:path_utils",
        "//ladder/core/common/logger",
    ],
)

make_py_checks(
    name = "video_utils_tests",
    data = [":mock_video_files"],
    test_srcs = glob(["test/*.py"]),
    test_timeout = "moderate",
    deps = [
        ":video_utils",
        "//ladder/core/common:path_utils",
    ],
)

filegroup(
    name = "mock_video_files",
    srcs = glob(["test/videos/**/*"]),
)
