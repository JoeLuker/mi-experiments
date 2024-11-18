from setuptools import setup, find_packages

setup(
    name="mlx-experiments",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "mlx",
        "dash",
        "plotly",
        "numpy",
        "pandas",
        "scikit-learn",
        "transformers",
        "huggingface_hub",
        "dash-bootstrap-components",
        "pytest",
    ],
    author="Joseph Luker",
    author_email="josephsluker@gmail.com",
    description="Mechanistic Interpretability Experiments",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)
