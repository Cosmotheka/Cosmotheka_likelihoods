from setuptools import find_packages, setup

setup(
    name="xCell_lkl",
    version="0.0",
    description="Cobaya likelihood for the xCell repository",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "cobaya>=3.0",
        "sacc>=0.4.2",
    ],
    package_data={"xCell_lkl": ["xCell_lkl.yaml"]},
)
