FROM amazon/aws-lambda-python:3.8

# install essential library
RUN yum update && yum install -y wget && yum clean all
RUN yum -y install cmake3 gcc gcc-c++ make && ln -s /usr/bin/cmake3 /usr/bin/cmake
RUN yum -y install python3-dev python3-setuptools libtinfo-dev zlib1g-dev build-essential libedit-dev llvm llvm-devel libxml2-dev git tar wget gcc gcc-c++

# git clone
RUN git clone https://github.com/jaeriver/TVM_Lambda_Container.git
# RUN git clone -b v0.8 --recursive https://github.com/apache/tvm tvm

# setup anaconda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda
# RUN cp tvm/conda/build-environment.yaml /tmp/build-environment.yaml
# RUN /opt/miniconda/bin/conda env create --file /tmp/build-environment.yaml --prefix /opt/conda-env
# RUN mv /var/lang/bin/python3.8 /var/lang/bin/python3.8-clean && ln -sf /opt/conda-env/bin/python /var/lang/bin/python3.8

# ENV PYTHONPATH "/var/lang/lib/python3.8/site-packages:/var/task"
ENV TVM_HOME=/var/task/TVM_Lambda_Container/tvm
ENV PATH=$PATH:$TVM_HOME/bin
ENV PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
ENV PATH=$TVM_HOME/python:$PATH

RUN pip3 install -r /var/task/TVM_Lambda_Container/requirements.txt

# install packages
WORKDIR TVM_Lambda_Container
RUN mkdir tvm/build
RUN cp /var/task/TVM_Lambda_Container/config.cmake tvm/build
RUN env CC=cc CXX=CC

WORKDIR tvm/build
RUN cmake ..
RUN make -j3


WORKDIR ../../

RUN cp /var/task/TVM_Lambda_Container/lambda_function.py ${LAMBDA_TASK_ROOT}
RUN cp -r /var/task/TVM_Lambda_Container/frozen_models /var/task/frozen_models
# RUN chmod 644 $(find . -type f)
# RUN chmod 755 $(find . -type d)

# ENTRYPOINT ["/lambda-entrypoint.sh"]
CMD ["lambda_function.lambda_handler"]
