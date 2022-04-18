from setuptools import setup

setup(
    name="lexenlem",
    version="0.0.1",
    packages=["lexenlem"],
    python_requires=">=3.6",
    install_requires=[
        "estnltk==1.7.0rc0",
        "numpy==1.21.2",
        "torch==1.6.0",
        "tqdm==4.64.0",
        "click==8.1.2",
        "regex==2022.3.15",
        "decorator==4.4.2",
        "conllu==4.4.1",
    ]
)
