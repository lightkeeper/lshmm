#!/usr/bin/env python
# Copyright (c) 2011 LightKeeper Advisors LLC
# ANY REDISTRIBUTION OR COPYING OF THIS MATERIAL WITHOUT THE EXPRESS CONSENT
# OF LIGHTKEEPER ADVISORS IS PROHIBITED.
# All rights reserved.
"""
Holds the setup information.  Please see the README for further setup
information.
"""
import os

from distribute_setup import use_setuptools
use_setuptools()
from setuptools import setup, find_packages, Extension

setup(name="lshmm",
      version="0.1dev",
      description = "LightStation Python HMM",
      author = "LightKeeper, LLC",
      author_email = "discuss-development@lightkeeper.com",
      url="http://www.lightkeeper.com",
      packages = find_packages('lib', exclude=['*.tests','*.tests.*', 'tests.*', 'tests']),
      package_dir = {'':'lib'},
      install_requires = ['numpy'],
      tests_require = ['nose >= 0.11', 'fudge >= 0.9'],
      ext_modules = [Extension("lshmm.libviterbi",
                               [os.path.join("lib", "lshmm", "viterbi.cc")],
                               extra_compile_args = ["-Wall", "-O2", "-ansi", "-pedantic"])],
      test_suite = "nose.collector",
      )


