from setuptools import setup, find_packages

setup(
    name='QuantGaussVector',
    version='1.0.0',
    author='Manav Gupta',
    author_email='manav26102002@gmail.com',
    description='A deep learning library for Gaussian Process Regression with quantization and vectorization.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MM21B038/QuantGaussVector',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'scipy>=1.7',
        'joblib>=1.0',
    ],
    extras_require={
        'dev': [
            'jupyter',
            'ipython',
            'pytest',
            'flake8',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)