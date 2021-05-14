from setuptools import find_packages, setup
from package import Package

setup(
    name="SINet",
    version="0.1",
    description="SINet CPU implementation",
    long_description=open('README.rst').read(),
    author="Samridha Shrestha",
    author_email="samridha.shrestha@g42.ai",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    cmdclass={
        "package": Package
    },
    package_data={'': ['weight/*.pth', ]},
    python_requires=">3.7.0"
)
