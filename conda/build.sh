#!/bin/bash

echo $PKG_VERSION > .version
# echo
# echo $PKG_VERSION
# echo
# echo $RECIPE_DIR
# echo
# echo $SRC_DIR
# echo
# echo $PREFIX
# echo
# echo $BUILD_PREFIX
env
ls $SRC_DIR
ls $RECIPE_DIR
$PYTHON $RECIPE_DIR/setup.py install --single-version-externally-managed --record record.txt

if [[ $(uname -o) != Msys ]]; then
  rm -rf "$SP_DIR/conda/shell/*.exe"
fi
$PYTHON -m conda init --install
if [[ $(uname -o) == Msys ]]; then
  sed -i "s|CONDA_EXE=.*|CONDA_EXE=\'${PREFIXW//\\/\\\\}\\\\Scripts\\\\conda.exe\'|g" $PREFIX/etc/profile.d/conda.sh
fi
