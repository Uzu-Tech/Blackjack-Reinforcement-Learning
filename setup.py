from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "src.game",
        ["src/game.cpp"],
        depends=["src/game.hpp"],
    ),
]

setup(
    name="game",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)