FROM amazon/aws-lambda-python:3.8


# # optional : ensure that pip is up to data
# RUN /var/lang/bin/python3.8 -m pip install --upgrade pip

# install essential library
RUN yum -y update
RUN yum -y install cmake3 gcc gcc-c++ make && ln -s /usr/bin/cmake3 /usr/bin/cmake
RUN yum -y install python3-dev python3-setuptools libtinfo-dev zlib1g-dev build-essential libedit-dev llvm llvm-devel libxml2-dev git tar wget gcc gcc-c++

# git clone
RUN git clone https://github.com/manchann/TVM_Lambda_Container.git

RUN git clone -b v0.8 --recursive https://github.com/apache/tvm tvm
ENV TVM_HOME=/tmp/tvm
ENV PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

RUN pip3 install -r /var/task/TVM_Lambda_Container/requirements.txt

# install packages
RUN mkdir tvm/build
RUN cp /var/task/TVM_Lambda_Container/config.cmake tvm/build
RUN env CC=cc CXX=CC

WORKDIR tvm/build
RUN cmake ..
RUN make -j3

WORKDIR ../
RUN rm -rf .git/

RUN cp /var/task/TVM_Lambda_Container/lambda_function.py /var/task/

CMD ["lambda_function.lambda_handler"]
