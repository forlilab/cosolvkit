#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Utils functions
#

import os
import sys

if sys.version_info >= (3, ):
    import importlib
else:
    import imp


def path_module(module_name):
    try:
        specs = importlib.machinery.PathFinder().find_spec(module_name)

        if specs is not None:
            return specs.submodule_search_locations[0]
    except:
        try:
            _, path, _ = imp.find_module(module_name)
            abspath = os.path.abspath(path)
            return abspath
        except ImportError:
            return None

    return None
