'''.

Module-wide configuration.

This is such an ugly hack of a way to accomplish this. Basically, I don't know
why complex/quaternion-valued tensors help learning in the case of the (much
simpler) neural latch, and I want an easy way to configure this
module-wide. After proper experimentation, this should be removed, and a number
system settled upon.

'''

import neurallambda.hypercomplex as H

##########
# Number system: Real, Complex, Quaternion
#
#   Why? in experiments on the Neurallatch, real-valued vectors could not learn
#   via backprop, but Complex- and Quaternion-valued vectors could.

# Number = H.Real
Number = H.Complex
# Number = H.Quaternion
N = Number # handy shorthand
