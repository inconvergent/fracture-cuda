# Fracture CUDA

CUDA implementation of https://github.com/inconvergent/fracture


![img](/img/img2.png?raw=true "img")

![img](/img/img.png?raw=true "img")

![img](/img/img3.png?raw=true "img")

![img](/img/img4.png?raw=true "img")

![ani](/img/ani.gif?raw=true "ani")

![ani](/img/ani2.gif?raw=true "ani")


## Prerequisites

This code relies on Python 3.

In order for this code to run you must first download and install:

  *    `iutils`: https://github.com/inconvergent/iutils
  *    `fn`: https://github.com/inconvergent/fn-python3

## Other Dependencies

The code also depends on:

  *    `numpy`
  *    `python-cairo` (do not install with pip, this generally does not work)
  *    `pycuda`

## TODO

  * Option to ignore visited sources when calculating steps
  * Fracture speed variation
  * Fracture intensity variation
  * Random nummers in kernel
  * Calculate Steps in kernel

