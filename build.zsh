export CFLAGS="-Wno-incompatible-function-pointer-types -Wno-incompatible-pointer-types"
export CXXFLAGS="$CFLAGS"
python setup.py bdist_wheel
