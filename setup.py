import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'requirements.txt')) as f:
    requires = f.read()


setup(
    name='openai-tests',
    version='0.1.0',
    description='openai-tests',
    packages=find_packages(),
    install_requires=requires,
    entry_points={
        'console_scripts': [
        ],
    },
)
