import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mesh-to-sdf",
    version="0.0.8",
    author="Marian Kleineberg",
    author_email="mail@marian42.de",
    description="Calculate signed distance fields for arbitrary meshes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marian42/mesh_to_sdf",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    data_files=[('shaders', ['mesh_to_sdf/shaders/mesh.frag', 'mesh_to_sdf/shaders/mesh.vert'])],
    include_package_data = True
)