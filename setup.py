from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="eitx",  # Required
    
    version="1.0.0",  # Required
    
    description="Fenicsx based library for the Electrical Impedance Tomography problem",  # Optional
    
    url="https://github.com/felipekriffel/EITx",  # Optional
    
    author="Felipe Kaminsky Riffel",  # Optional
    
    # package_dir={"": "eitx"},  # Optional
    # packages=find_packages(where="eitx"),  # Required
    packages=['eitx'],

    python_requires=">=3.10, <4",
)