from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

version = '0.0.1'

setup(name='WrapML',
      packages=find_packages(),
      version=version,
      description='Simplifying ML.',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/bugo99iot/wrapml',
      author='Ugo Bee',
      author_email='ugo.bertello@gmail.com',
      license='MIT',
      keywords=['machine learning', 'deep learning'],
      python_requires='>=3.6',
      classifiers=['Programming Language :: Python :: 3.6'],
      install_requires=required,
      include_package_data=True,
      zip_safe=False
      )
