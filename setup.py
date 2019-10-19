import os
import sys

from setuptools import find_packages, setup

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Find version
for line in open(os.path.join(PROJECT_PATH, 'pyroapi', 'version.py')):
    if line.startswith('__version__ = '):
        version = line.strip().split()[2][1:-1]

# READ README.md for long description on PyPi.
try:
    long_description = open('README.md', encoding='utf-8').read()
except Exception as e:
    sys.stderr.write('Failed to convert README.md to rst:\n  {}\n'.format(e))
    sys.stderr.flush()
    long_description = ''


setup(
    name='pyro-api',
    version=version,
    description='Generic API for dispatch to Pyro backends.',
    packages=find_packages(include=['pyroapi', 'pyroapi.*']),
    url='https://github.com/pyro-ppl/pyro-api',
    author='Uber AI Labs',
    author_email='npradhan@uber.com',
    install_requires=[],
    extras_require={
        # PyPi does not like @ versions,
        # so I comment out the 'test' section when uploading to pypi.
        'test': [
            'flake8',
            'pytest>=5.0',
            'pyro-ppl@https://api.github.com/repos/pyro-ppl/pyro/tarball/dev',
            'numpyro@https://api.github.com/repos/pyro-ppl/numpyro/tarball/master',
            'funsor@https://api.github.com/repos/pyro-ppl/funsor/tarball/master',
        ],
        'dev': [
            'sphinx>=2.0',
            'sphinx_rtd_theme',
            'ipython',
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    tests_require=['flake8', 'pytest>=4.1'],
    keywords='probabilistic machine learning bayesian statistics',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
    ],
)
