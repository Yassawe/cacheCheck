from ctypes import *
cdll.LoadLibrary('libcupti.so.10.1')
libc = CDLL('libcupti.so.10.1')

print(libc)