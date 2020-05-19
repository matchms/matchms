"%PYTHON%" -m pip install --no-deps --ignore-installed . -vv

mkdir %PREFIX%\site-packages\matchms\data
copy matchms\data\* %PREFIX%\site-packages\matchms\data\

if errorlevel 1 exit 1
