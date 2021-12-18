from ctypes import *
dll = cdll.LoadLibrary('./executables/hardware_counter.so')

dll.main()