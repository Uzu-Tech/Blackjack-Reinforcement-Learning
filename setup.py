from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "blackjack_env", 
        ["src/main.cpp", "src/hand.cpp"],
        depends=["src/main.hpp", "src/hand.hpp"],
        extra_compile_args=[
            '/O2',
            '/DNDEBUG',
            '/GL',
            '/fp:fast',
            '/Ob3',
            '/Ot',
            '/Oi'
        ],
        extra_link_args=['/LTCG']
    ),
]

setup(
    name="blackjack_nev",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)