import importlib
import inspect
import os
import sys


def list_classes_and_functions(package_name):
    results = {}
    package = importlib.import_module(package_name)
    package_path = package.__path__[0]

    def walk_package(package_path, package_name):
        for root, _, files in os.walk(package_path):
            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    module_path = os.path.join(root, file)
                    relative_module_path = os.path.relpath(module_path, package_path)
                    module_name = os.path.splitext(
                        relative_module_path.replace(os.sep, ".")
                    )[0]
                    full_module_name = f"{package_name}.{module_name}"

                    try:
                        module = importlib.import_module(full_module_name)
                        results[full_module_name] = {"classes": {}, "functions": []}

                        for name, obj in inspect.getmembers(module):
                            if (
                                inspect.isclass(obj)
                                and obj.__module__ == full_module_name
                            ):
                                class_methods = [
                                    method_name
                                    for method_name, method_obj in inspect.getmembers(
                                        obj
                                    )
                                    if inspect.isfunction(method_obj)
                                ]
                                results[full_module_name]["classes"][
                                    name
                                ] = class_methods
                            elif (
                                inspect.isfunction(obj)
                                and obj.__module__ == full_module_name
                            ):
                                results[full_module_name]["functions"].append(name)
                    except ImportError as e:
                        print(
                            f"Failed to import module {full_module_name}: {e}",
                            file=sys.stderr,
                        )

    walk_package(package_path, package_name)
    return results


def write_results_to_file(results, output_file):
    with open(output_file, "w") as f:
        f.write(".. autosummary::\n   :toctree:\n   :recursive:\n   :nosignatures:\n\n")

        for module, members in results.items():
            for cls, methods in members["classes"].items():
                for method in methods:
                    if method == "__init__":
                        f.write(f"    {module}.{cls}\n")
                        f.write(f"    {module}.{cls}.{method}\n")
                    elif method == "__repr__":
                        f.write(f"    {module}.{cls}\n")
                    else:
                        f.write(f"    {module}.{cls}.{method}\n")
            for func in members["functions"]:
                f.write(f"    {module}.{func}\n")


result = list_classes_and_functions("lenapy")
write_results_to_file(result, os.path.join(os.getcwd(), "functions_classes.rst"))
