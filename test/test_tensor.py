'''

Test neurallambda.tensor

'''

import neurallambda.tensor as T
from neurallambda.tensor import Neurallambda
import neurallambda.hypercomplex as H

nl = Neurallambda(
    n_addresses=2,
    vec_size=128,
    n_stack=2,
    gc_steps=0,
    number_system=H.Real,
    device='cpu',
)

def test_int_projections():
    # Round trip: char->int->vec->int->char
    for c in T.chars:
        d = T.int_to_char(nl.unproject_int(nl.project_int(T.char_to_int(c))))
        assert c == d

def test_tag_projections():
    for expected_ix, tag in enumerate(T.tag_names):
        assert T.tag_names[expected_ix] == nl.vec_to_tag(nl.tag_to_vec[tag])
    print('done')
