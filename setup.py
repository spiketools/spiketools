"""spiketools setup script."""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('spiketools', 'version.py')) as version_file:
    exec(version_file.read())

# Load the long description from the README
with open('README.rst') as readme_file:
    long_description = readme_file.read()

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name = 'spiketools',
    version = __version__,
    description = 'Module for analyzing spiking data.',
    long_description = long_description,
    long_description_content_type = 'text/x-rst',
    python_requires = '>=3.6',
    maintainer = 'Thomas Donoghue',
    maintainer_email = 'tdonoghue.research@gmail.com',
    url = 'https://github.com/spiketools/spiketools',
    packages = find_packages(),
    license = 'Apache License, 2.0',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    platforms = 'any',
    project_urls = {
        'Documentation' : 'https://spiketools.github.io/',
        'Bug Reports' : 'https://github.com/spiketools/spike/issues',
        'Source' : 'https://github.com/spiketools/spiketools'
    },
    download_url = 'https://github.com/spiketools/spiketools/releases',
    keywords = ['neuroscience', 'single units', 'spike analyses', 'electrophysiology'],
    install_requires = install_requires,
    extras_require = {
        "stats" : ["statsmodels"],
    },
    tests_require = ['pytest'],
)
