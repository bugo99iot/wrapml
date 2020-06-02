import pathlib
import pkg_resources
from setuptools import setup, find_packages

# todo: remove unneeded packages such as tensorboard, twine, etc
with pathlib.Path('requirements.txt').open() as requirements_txt:
    required = [str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)]

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

version = '0.0.15'

setup(name='wrapml',
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
