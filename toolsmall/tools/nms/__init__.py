from .nms import nms,nmsV2,soft_nms,diouNms
from .py_cpu_nms import py_cpu_nms

# Pyximport 直接导入pyx
# import pyximport; pyximport.install()
# import soft_nms
# from .soft_nms import soft_nms