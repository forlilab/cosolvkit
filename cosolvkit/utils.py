#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Utils functions
#

import contextlib
import os
import tempfile
import shutil
import subprocess
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


@contextlib.contextmanager
def temporary_directory(suffix=None, prefix=None, dir=None):
    """Create and enter a temporary directory; used as context manager."""
    temp_dir = tempfile.mkdtemp(suffix, prefix, dir)
    cwd = os.getcwd()
    os.chdir(temp_dir)
    try:
        yield temp_dir
    finally:
        os.chdir(cwd)
        shutil.rmtree(temp_dir)


def execute_command(cmd_line):
    """Simple function to execute bash command."""
    args = cmd_line.split()
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, errors = p.communicate()
    return output, errors

