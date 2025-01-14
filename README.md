# cyclops
## A monocular VIO library with very fast and robust scale initialization
**Cyclops** is a software library for monocular visual-inertial odometry (MVIO)
of a camera-IMU system. This software library is supplementary material for an
article we submitted for review to IEEE Transactions on Robotics.

Cyclops features very fast and accurate initialization compared to other MVIO
algorithms, e.g. [ORB-SLAM 3](https://github.com/UZ-SLAMLab/ORB_SLAM3) and
[VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono). This fast
initialization is based on almost-exact 1D approximation of MVIO optimization,
as detailed in our submitted paper.

# Installation
Tested on Ubuntu 20.04.

## Quick start
``` bash
$ sudo apt-get update && \
  sudo apt-get install -y libgflags-dev libgoogle-glog-dev
$ cmake -S. -B./build/Release \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install \
  -Dcyclops_native_build=yes
$ cmake --build ./build/Release --target install -- -j$(nproc)
```

## Required dependencies
* [Glog](https://github.com/google/glog)
* [Gflags](https://github.com/gflags/gflags)

## Bundled dependencies
The following libraries are automatically configured and locally installed in the
build directory of Cyclops. If you want to use externally installed versions of
these libraries (e.g. in `/usr/local`, or elsewhere specified by adding a cmake
argument `-DCMAKE_PREFIX_PATH=<...>`), append the following argument to the CMake
configuration command,
``` bash
$ cmake -S. -B./build/Release <...> -Dcyclops_configure_bundle_dependencies=no
```
* [Ceres solver](https://github.com/ceres-solver/ceres-solver)
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
* [Range-v3](https://github.com/ericniebler/range-v3)
* [Spdlog](https://github.com/gabime/spdlog)

## Test dependencies
The following libraries are required for unit testing, which can be enabled by
`-Dcyclops_test=yes`. Similar to the bundled libraries, the automatic
configuration of these libraries is controlled by appending:
``` bash
$ cmake -S. -B./build/Release <...> -Dcyclops_configure_test_dependencies=no
```
* [nlohmann/json](https://github.com/nlohmann/json)
* [Doctest](https://github.com/doctest/doctest)

Additionally, some test cases depend on [Boost](https://www.boost.org/). These
additional test cases are disabled by default and can be enabled by
``` bash
$ cmake -S. -B./build/Release <...> -Dcyclops_test_with_boost=yes
```

# Usage
See [cyclops_ros](https://github.com/cyclops-double-blind/cyclops_ros) for the
ROS wrapper of this library. Also, visit
[cyclops_playground](https://github.com/cyclops-double-blind/cyclops_playground)
for a usage example of our VIO algorithm on the EuRoC-MAV dataset.
