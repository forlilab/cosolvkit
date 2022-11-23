#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import fnmatch
from setuptools import setup, find_packages


def find_files(directory):
    matches = []

    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))

    return matches


setup(name="cosolvkit",
      version='0.3',
      description="CoSolvKit",
      author="Jerome Eberhardt",
      author_email="jerome@scripps.edu",
      url="https://github.com/jeeberhardt/cosolvkit",
      packages=find_packages(),
      scripts=["scripts/wk_prepare_receptor.py"],
      package_data={
            "cosolvkit" : ["data/*"]
      },
      data_files=[("", ["README.md", "LICENSE"]),
                  ("scripts", find_files("scripts"))],
      include_package_data=True,
      zip_safe=False,
      license="MIT",
      keywords=["molecular modeling", "drug design",
                "cosolvent", "MD simulations"],
      classifiers=["Programming Language :: Python :: 3.7",
                   "Operating System :: Unix",
                   "Operating System :: MacOS",
                   "Topic :: Scientific/Engineering"]
)
