from setuptools import setup, find_packages

setup(
    name='ledidi',
    version='0.0.1',
    author='Yang Lu and Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['ledidi'],
    package_dir={'': 'src'},
    scripts=['ledidi'],
    url='http://pypi.python.org/pypi/ledidi/',
    license='LICENSE.txt',
    description='Ledidi is an optimization approach for designing edits to biological sequences.'
)