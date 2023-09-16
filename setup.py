#!/usr/bin/env python

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="loopgen",
        version="0.0.1",
        packages=find_packages(),
        authors="Matt Greenig",
        email="mg989@cam.ac.uk",
        description="LoopGen: De novo design of peptide CDR binding loops with SE(3) diffusion models.",
        include_package_data=True,
        entry_points={"console_scripts": ["loopgen=loopgen.__main__:main"]},
    )
