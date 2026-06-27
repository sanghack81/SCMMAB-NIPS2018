from setuptools import setup, find_packages

setup(
    name='npsem',
    version="0.1.0",
    packages=find_packages(),
    author='Sanghack Lee',
    author_email='sanghack.lee@gmail.com',
    install_requires=[
        'numpy',
        'scipy',
        'joblib',
        'matplotlib',
        'seaborn',
        'networkx'
    ],
    extras_require={
        'test': ['pytest'],
    },
    python_requires='>=3.11',
)
