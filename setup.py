from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(this_directory, "requirements.txt")) as f:
    requirements = f.read().splitlines()


setup(
    name="aivault",
    version="0.0.1",
    description="AIVault",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Josh XT",
    author_email="josh@devxt.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
)
