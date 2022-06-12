import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="ShapFire",
    version="0.1.0",
    author="NicklasXYZ",
    author_email="",
    description="",
    # long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/nicklasxyz/shapfire",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
