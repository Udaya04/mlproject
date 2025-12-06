from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    requirements= []
    with open(file_path,"r") as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements
setup(
    name="Mlprojects",
    version="0.1",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    author="Udayasankar Sahoo",
    author_email="sahooudayasankar99@gmail.com",
    description="machine learning projects.",
)

