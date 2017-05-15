import os
import setuptools

setuptools.setup(
    name='pthbldr',
    version='0.0.1',
    packages=setuptools.find_packages(),
    author='Kyle Kastner',
    author_email='kastnerkyle@gmail.com',
    description='Deep Learning tools for PyTorch',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.rst')).read(),
    license='BSD 3-clause',
    url='http://github.com/kastnerkyle/pthbldr/',
    install_requires=['numpy',
                      'scipy',
                      'torch'],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering'],
)
