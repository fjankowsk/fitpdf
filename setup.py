import os.path
from setuptools import setup


def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "fitpdf", "version.py")

    with open(version_file, "r") as f:
        raw = f.read()

    items = {}
    exec(raw, None, items)

    return items["__version__"]


setup(
    version=get_version(),
)
