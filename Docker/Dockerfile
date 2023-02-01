FROM gsjung/al-ornl
RUN apt-get update
RUN apt-get install -y vim 
ENV MKL_THREADING_LAYER=GNU
ENV LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/lib:$LD_LIBRARY_PATH
#RUN pip install -U numpy
