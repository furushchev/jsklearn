#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

try:
    # for ROS
    from distutils.core import setup
    from catkin_pkg.python_setup import generate_distutils_setup

    d = generate_distutils_setup(
        packages=['jsklearn'],
        package_dir={'':'src'}
    )
    setup(**d)

except:
    # for Python
    from setuptools import find_packages
    from setuptools import setup
    import xml.etree.ElementTree as ET

    pkg = ET.parse("package.xml").getroot()

    setup(
        name=pkg.findtext("name"),
        description=pkg.findtext("description"),
        long_description=open("README.md").read(),
        version=pkg.findtext("version"),
        author=pkg.findtext("author"),
        author_email=pkg.find("author").get("email"),
        url=pkg.findtext("url"),
        license=pkg.findtext("license"),
        packages=find_packages(),
        install_requires=open('requirements.txt').readlines(),
    )
