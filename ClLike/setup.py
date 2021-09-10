from setuptools import find_packages, setup

setup(
    name="cl_like",
    version="0.0",
    description="Multi-tracer angular C_ell likelihood",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "cobaya>=3.0",
        "sacc>=0.4.2",
        "pyccl",
        "numpy",
        "scipy"
    ],
    package_data={"cl_like": ["ClLike.yaml"]},
)
