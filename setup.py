import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DataAnonynmizer",
    version="0.1",
    author="Kelvin Tay",
    author_email="btkelvin@gmail.com",
    description="Anonymizes pandas dataset and provides a hash dictionary to de-anonymize",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    license="MIT license",
    install_requires=[
        "transformers>=3.0.0",
        "torch>=1.5.0",
        "torchvision>=0.6.0"
    ],
    python_requires=">=3.6.0"
)