REM remove build artifacts

DEL /s /q dist
DEL /s /q wheelhouse
DEL /s /q SINet.egg-info
DEL /s /q __pycache__

RMDIR /s /q dist
RMDIR /s /q wheelhouse
RMDIR /s /q SINet.egg-info
RMDIR /s /q __pycache__
