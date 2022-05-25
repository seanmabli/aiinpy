from setuptools import setup, Extension

setup(
    name='hello-lib',
    version='1',
    ext_modules=[Extension('_hello', ['_hello.c'])],
)