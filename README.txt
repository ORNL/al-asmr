# README

cd Docker && docker build . -t activeml && cd ..
docker run -it activeml /bin/bash
docker run -it activeml python ./schnet.py
docker run -it -v "$(pwd)"/AL:/workspace activeml /bin/bash
docker run -it -v "$(pwd)"/AL_dftb:/workspace activeml /bin/bash
docker run -it -v "$(pwd)"/AL_schnet:/workspace activeml /bin/bash
