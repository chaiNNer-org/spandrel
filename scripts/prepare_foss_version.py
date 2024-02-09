"""
    This script removes non-FOSS compliant code from the architectures directory and modifies the repo to be able to publish a different package.
"""

import shutil

import toml

archs_to_remove = ["CodeFormer", "DDColor", "FeMaSR", "GFPGAN", "MAT", "SRFormer"]

# Remove non-FOSS compliant code
for arch in archs_to_remove:
    shutil.rmtree(f"src/spandrel/architectures/{arch}", ignore_errors=True)

# Change pyproject.toml info

# Read in the file
with open("pyproject.toml") as file:
    data = toml.load(file)

note_string = "This version of Spandrel is FOSS compliant as it remove support for model architectures that are under a non-commercial license."

# Change the package name
data["project"]["name"] = "spandrel-foss"
data["project"]["description"] = f'{data["project"]["description"]} {note_string}'

# Write the file out again
with open("pyproject.toml", "w") as file:
    toml.dump(data, file)


# Update the readme
with open("README.md") as file:
    readme = file.read()

readme_split = readme.split("\n")
readme = "\n".join([readme_split[0]] + [note_string] + readme_split[1:])

with open("README.md", "w") as file:
    file.write(readme)


# Update the registry
with open("src/spandrel/__helpers/main_registry.py") as file:
    registry = file.read()

for arch in archs_to_remove:
    registry = registry.replace(f"    {arch},\n", "")
    registry_split = registry.split("\n")
    registry_out = registry_split
    for i, line in enumerate(registry_split):
        if f'id="{arch}"' in line:
            while "ArchSupport" not in registry_split[i]:
                registry_out[i] = ""
                i -= 1
            registry_out[i] = ""
            i += 1
            while "ArchSupport" not in registry_split[i]:
                registry_out[i] = ""
                i += 1
    registry = "\n".join(registry_out)

with open("src/spandrel/__helpers/main_registry.py", "w") as file:
    file.write(registry)
