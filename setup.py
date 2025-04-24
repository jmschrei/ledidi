from setuptools import setup, find_packages

setup(
    name='ledidi',
    version='2.1.0',
    author='Jacob Schreiber and Yang Lu',
    author_email='jmschreiber91@gmail.com',
    packages=['ledidi'],
    url='http://pypi.python.org/pypi/ledidi/',
    license='MIT License',
    description='Ledidi is an optimization approach for designing edits to biological sequences.',
    install_requires=[
        "torch >= 1.9.0",
        "matplotlib"
    ],
)