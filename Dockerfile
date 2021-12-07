FROM amazon/aws-lambda-python:3.8

# optional : ensure that pip is up to data
RUN /var/lang/bin/python3.8 -m pip install --upgrade pip

# install essential library
RUN yum -y update
RUN yum -y install cmake3 gcc gcc-c++ make && ln -s /usr/bin/cmake3 /usr/bin/cmake
RUN yum -y install python3-dev python3-setuptools libtinfo-dev zlib1g-dev build-essential libedit-dev libxml2-dev git tar wget gcc gcc-c++

# git clone
RUN git clone https://github.com/manchann/TVM_Lambda_Container.git

WORKDIR TVM_Lambda_Container
RUN pip3 install --user -r TVM_Lambda_Container/requirements.txt
RUN git clone -b v0.8 --recursive https://github.com/apache/tvm tvm
# install packages
RUN mkdir tvm/build
RUN cp config.cmake tvm/build
RUN env CC=cc CXX=CC

RUN export TVM_HOME=~/TVM_Lambda_Container/tvm
RUN export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

WORKDIR tvm/build
RUN cmake ..
RUN make -j4

CMD ["lambda_function.lambda_handler"]
