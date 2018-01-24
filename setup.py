from setuptools import setup, find_packages

setup(
    name='tuner',
    version='0.0.1',
    description='auto tuning machiner learning',
    license='MIT',
    author='FUKUDA Yutaro',
    url='https://github.com/sktnkysh/tuner',
    packages=find_packages(),
    entry_points={
        'console_scripts':
        ['format-dataset=tuner.scripts.format_dataset:main', 'toon=tuner.scripts.toon:main']
    },
)
