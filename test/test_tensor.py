'''

Test neurallambda.tensor

'''

import neurallambda.tensor as T
import neurallambda.symbol as S

nl = T.build_empty_neurallambda(
    batch_size=1,
    n_addresses=2,
    vec_size=512,
    zero_vec_bias=1e-3,
    device='cpu',
)

def test_int_projections():
    ''' Round trip: char -> int -> vec -> int -> char '''
    for c in S.chars:
        d = S.int_to_char(nl.unproject_int(nl.project_int(S.char_to_int(c))))
        assert c == d

def test_tag_projections():
    for expected_ix, tag in enumerate(T.tag_names):
        assert T.tag_names[expected_ix] == nl.vec_to_tag(nl.tag_to_vec[tag])
