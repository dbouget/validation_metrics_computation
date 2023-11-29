from setuptools import find_packages, setup
import platform
import sys

with open("README.md", "r", errors='ignore') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-16', errors='ignore') as ff:
    required = ff.read().splitlines()

setup(
    name='raidionicsval',
    packages=find_packages(
        include=[
            'raidionicsval',
            'raidionicsval.Computation',
            'raidionicsval.Validation',
            'raidionicsval.Plotting',
            'raidionicsval.Studies',
            'raidionicsval.Utils',
            'tests',
        ]
    ),
    entry_points={
        'console_scripts': [
            'raidionicsval = raidionicsval.__main__:main'
        ]
    },
    install_requires=required,
    include_package_data=True,
    python_requires=">=3.8",
    version='1.0.0',
    author='David Bouget (david.bouget@sintef.no)',
    license='BSD 2-Clause',
    description='Raidionics backend for running validation and metrics computation',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
