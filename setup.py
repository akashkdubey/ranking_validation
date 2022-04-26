from setuptools import setup

setup(
    name="rank validation",
    version="0.0.1",
    description="create a rank validation report",
    py_modules=["validation_generator", "metrics", "normalise"],
    package_dir={'': 'src'},
    author="Akash Dubey",
    author_email="akashdubey826@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ]
)
