# -*- coding: utf-8 -*-
import os, sys
from setuptools import setup, find_packages

def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='chokozainerrl',
<<<<<<< HEAD
    version='0.0.61',
=======
    version='0.0.60.17',
>>>>>>> 9616a4b67b573abdfa2b5d2fe42ae24fb3253fac
    description='Wrapper package for chainerRL',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='chokozainer',
    author_email='fgtohru@gmail.com',
    install_requires=read_requirements(),
    url='https://github.com/chokozainer/chokozainerrl',
    license=license,
    packages=find_packages(exclude=('tests', 'docs','sample','notebook'))
)

