from setuptools import setup


setup(name='multidocsum', 
      version='0.0.1',
      packages=['src',
                'src.data',
                'src.features',
                'src.models'],
      python_requires='==3.7.*', 
      install_requires=['nltk==3.5',
                        'numpy==1.19.4',
                        'regex==2020.11.13',
                        'scikit-learn==0.21.3',
                        'scipy==1.5.4',
                        'underthesea==1.2.2',
                        'pyvi==0.1',
                        'pandas==1.1.4'])