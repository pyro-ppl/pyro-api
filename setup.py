import os
import sys

from setuptools import find_packages, setup

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Find version
for line in open(os.path.join(PROJECT_PATH, 'pyro_api', 'version.py')):
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
    description='Pyro API for generic model dispatch.',
    packages=find_packages(include=['pyro_api', 'pyro_api.*']),
    url='https://github.com/pyro-ppl/pyro-api',
    author='Uber AI Labs',
    author_email='npradhan@uber.com',
    install_requires=[],
    # TODO: Change the dependency versions as needed.
    extras_require={
        'test': [
            'flake8',
            'pytest>=5.0',
            'pyro-ppl@https://api.github.com/repos/pyro-ppl/pyro/tarball/7caad72',
            'numpyro@https://api.github.com/repos/pyro-ppl/numpyro/tarball/7174846',
            'funsor@https://api.github.com/repos/pyro-ppl/funsor/tarball/3d6197f',
        ],
        'dev': ['ipython'],
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