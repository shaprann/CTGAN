import importlib
import sys

modules = {
    "CTGAN": "../__init__.py",
    "spatiotemporal": "../spatiotemporal/__init__.py"
}

for module_name, module_location in modules.items():

    spec = importlib.util.spec_from_file_location(module_name, module_location)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)