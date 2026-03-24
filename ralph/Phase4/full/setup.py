from setuptools import setup, find_packages
import os

# Resolve absolute paths to local phase packages so pip can find them
_here = os.path.abspath(os.path.dirname(__file__))
_phase1 = os.path.normpath(os.path.join(_here, "../../Phase1/006"))
_phase2 = os.path.normpath(os.path.join(_here, "../../Phase2/full"))
_phase3 = os.path.normpath(os.path.join(_here, "../../Phase3/full"))

setup(
    name="rlqas",
    version="4.0.0",
    description="Reinforcement Learning Quantum Architecture Search — unified package",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        f"rlqas-phase1 @ file://{_phase1}",
        f"rlqas-phase2 @ file://{_phase2}",
        f"rlqas-phase3 @ file://{_phase3}",
    ],
    entry_points={
        "console_scripts": [
            "rlqas=rlqas.cli:main",
        ],
    },
)
