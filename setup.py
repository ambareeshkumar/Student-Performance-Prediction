from setuptools import find_packages, setup
from typing import List

Hypen_e_dot = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    :param file_path:
    :return:  return list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n","") for req in requirements]

        if Hypen_e_dot in requirements:
            requirements.remove(Hypen_e_dot)

setup (
    name = "student_performance_pred",
    version = '0.0.1',
    author = "Ambareeshkumar",
    author_email = "ambareeshkumar10@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)