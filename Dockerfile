FROM amazon/aws-lambda-python:3.8

# optional : ensure that pip is up to data
RUN /var/lang/bin/python3.8 -m pip install --upgrade pip

# install essential library
RUN yum -y update
RUN yum install python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential libedit-dev libxml2-dev git tar wget gcc-c++ -y
RUN mkdir -p /tmp/cmake \
 &&  cd /tmp/cmake \
 && curl -Ls  https://github.com/Kitware/CMake/releases/download/v3.15.3/cmake-3.15.3.tar.gz | tar xzC /tmp/cmake --strip-components=1 \
 && ./bootstrap --prefix=/usr/local \
 && make \
 && make install
 
# git clone
RUN git clone https://github.com/manchann/TVM_Lambda_Container.git

# install packages
RUN pip install --user -r TVM_Lambda_Container/requirements.txt

WORKDIR TVM_Lambda_Container
RUN mkdir tvm/build
RUN cp config.cmake tvm/build

RUN export TVM_HOME=~/TVM_Lambda_Container/tvm
RUN export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

WORKDIR tvm/build
RUN cmake3 ..
RUN make -j4

CMD ["lambda_function.lambda_handler"]
