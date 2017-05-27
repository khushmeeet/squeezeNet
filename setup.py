from distutils.core import setup
import os.path
import codecs
import re


def fpath(name):
    return os.path.join(os.path.dirname(__file__), name)


def read(fname):
    return codecs.open(fpath(fname), encoding='utf-8').read()

def grep(attrname):
    pattern = r"{0}\W*=\W*'([^']+)'".format(attrname)
    strval, = re.findall(pattern, file_text)
    return strval


file_text = read(fpath('squeezenet/__init__.py'))

setup(
  name='squeezenet',
  packages=['squeezenet'],
  version=grep('__version__'),
  description='A convolutional neural network with SqueezeNet architecture',
  author='Khushmeet Singh',
  author_email='khushmeetsingh199@gmail.com',
  url='https://github.com/Khushmeet/squeezeNet',
  download_url='https://github.com/Khushmeet/squeezeNet/tarball/1.0',
  keywords=['Deep learning', 'Tensorflow', 'Convolutional', 'Neural network', 'SqueezeNet'],
  install_requires=[
   'tensorflow'
  ],
  classifiers=[],
)