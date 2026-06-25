from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='birdcalledge',
    version='0.0.1',
    description='Neuromorphic bird sound detection.',
    long_description=readme,
    author='Daniel Müller-Komorowska',
    author_email='danielmuellermsc@gmail.com',
    url='https://github.com/danielmk/birdcalledge',
    license=license,
    packages=['birdcalledge'],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'librosa',
	    'tables',
        'sounddevice'],)
