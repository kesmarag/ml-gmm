from setuptools import setup

setup(name='kesmarag-ml-gmm',
      version='0.0.4',
      description='Gaussian Mixture Model',
      author='Costas Smaragdakis',
      author_email='kesmarag@gmail.com',
      url='https://github.com/kesmarag/ml-gmm',
      packages=['kesmarag.ml.gmm'],
      package_dir={'kesmarag.ml.gmm': './'},
      install_requires=['numpy', 'scipy', 'sklearn', 'matplotlib'], )