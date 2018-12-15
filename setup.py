import os
import codecs
import re
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
package_name = "treemodel"


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name=package_name,
    version=find_version("__version__.py"),

    description="Package provides functionality for modelling on tree-shaped data.",

    url="https://github.com/ryshoooo/treeModel",
    author="Richard Nemeth",
    author_email="ryshoooo@gmail.com",
    license="GNU GPL v3",

    keywords='machine learning tree modelling',

    install_requires=[
        'numpy==1.15.2',
    ],



    packages=find_packages(exclude=['test']),
)
