import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name='art_inspector',
    py_modules=['src'],
    version='1.0',
    description='',
    author='Amar',
    install_requires=[
        str(r) for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), 'requirements_dev.txt'))
        )
    ]
    # dependency_links=[
    #     'https://download.pytorch.org/whl/torch_stable.html'
    # ]
)
