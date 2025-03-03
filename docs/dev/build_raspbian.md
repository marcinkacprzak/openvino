# Build OpenVINO™ Runtime for Raspbian Stretch OS

> **NOTE**: Since 2023.0 release, you can compile [OpenVINO Intel CPU plugin](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_cpu) on ARM platforms.

## Hardware Requirements
* Raspberry Pi 2 or 3 with Raspbian Stretch OS (32 or 64-bit).

  > **NOTE**: Despite the Raspberry Pi CPU is ARMv8, 32-bit OS detects ARMv7 CPU instruction set. The default `gcc` compiler applies ARMv6 architecture flag for compatibility with lower versions of boards. For more information, run the `gcc -Q --help=target` command and refer to the description of the `-march=` option.

## Compilation
You can perform native compilation of the OpenVINO Runtime for Raspberry Pi, which is the most straightforward solution. However, it might take at least one hour to complete on Raspberry Pi 3.

1. Install dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install -y git cmake scons build-essential
  ```
2. Clone the repository:
```
git clone --recurse-submodules --single-branch --branch=master https://github.com/openvinotoolkit/openvino.git 
```
3. Go to the cloned `openvino` repository:

  ```bash
  cd openvino/
  ```
4. Create a build folder:

  ```bash
  mkdir build && cd build/
  ```
5. Build the OpenVINO Runtime:
  ```bash
  cmake -DCMAKE_BUILD_TYPE=Release \
        -DARM_COMPUTE_SCONS_JOBS=$(nproc --all) \
  .. && cmake --build . --parallel 
  ```

## Additional Build Options

- To build Python API, install `libpython3-dev:armhf` and `python3-pip`
  packages using `apt-get`; then install `numpy` and `cython` python modules
  via `pip3`, adding the following options:
   ```sh
   -DENABLE_PYTHON=ON \
   -DPYTHON_EXECUTABLE=/usr/bin/python3.7 \
   -DPYTHON_LIBRARY=/usr/lib/arm-linux-gnueabihf/libpython3.7m.so \
   -DPYTHON_INCLUDE_DIR=/usr/include/python3.7
   ```

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)

