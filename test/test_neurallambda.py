'''

Test neurallambda.tensor

'''

import torch
import neurallambda.neurallambda as N
import neurallambda.symbol as S
# from neurallambda.neurallambda import CosineSimilarity, Weight, ReverseCosineSimilarity
from neurallambda.torch import cosine_similarity

nl = N.build_empty_neurallambda(
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
    for expected_ix, tag in enumerate(N.tag_names):
        assert N.tag_names[expected_ix] == nl.vec_to_tag(nl.tag_to_vec[tag])


# ##################################################
# # Cosine Similarity

# def test_initialization_methods():
#     input_features, output_features = 10, 1024
#     for method in ['kaiming', 'xavier', 'orthogonal']:
#         w = Weight(input_features, output_features, init_method=method)
#         model = CosineSimilarity(w, dim=0)
#         assert model.weight.shape == (input_features, output_features), f"Weight shape does not match expected shape: {model.weight.shape}"

# def test_cosine_similarity_forward_shape():
#     input_features, output_features = 10, 1024
#     batch_size = 4
#     w = Weight(input_features, output_features)
#     model = CosineSimilarity(w, dim=1, unsqueeze_inputs=[2], unsqueeze_weights=[0])
#     input_tensor = torch.randn(batch_size, input_features)
#     output = model(input_tensor)
#     assert output.shape == (batch_size, output_features), f"Output shape does not match expected shape, got: {output.shape}"

# def test_cosine_similarity_for_identical_vectors():
#     n_vectors, vec_size = 13, 4096
#     batch_size = 7
#     w = Weight(n_vectors, vec_size)
#     model = CosineSimilarity(w, unsqueeze_inputs=[1], unsqueeze_weights=[0], dim=2)

#     inputs = w.weight.clone().detach()  # Clone the weight to use as an input
#     inputs = inputs[:batch_size]

#     output = model(inputs)
#     expected = torch.zeros((batch_size, n_vectors))
#     for i in range(batch_size):
#         expected[i,i] = 1.0

#     assert output.shape == expected.shape
#     assert torch.allclose(output, expected, atol=0.1), "Cosine similarity of identical vectors should be close to 1."

# def test_cosine_similarity_for_dissimilar_vectors():
#     n_vectors, vec_size = 13, 4096
#     batch_size = 7
#     w = Weight(n_vectors, vec_size)
#     model = CosineSimilarity(w, unsqueeze_inputs=[1], unsqueeze_weights=[0], dim=2)

#     inputs = torch.randn(batch_size, vec_size)

#     output = model(inputs)
#     expected = torch.zeros((batch_size, n_vectors))
#     assert output.shape == expected.shape, f'output: {output.shape}, expected: {expected.shape}'
#     assert torch.allclose(output, expected, atol=0.1), "Cosine similarity of identical vectors should be close to 1."


# def test_cosine_similarity_with_one_input():
#     input_features, output_features = 1024, 256
#     w = Weight(input_features, output_features, init_method='kaiming')
#     model = CosineSimilarity(w, dim=1)
#     input_tensor = torch.randn(1, output_features)
#     output = model(input_tensor)
#     assert output.shape == torch.Size([1024]), f'Wrong shape: {output.shape}'

# def test_cosine_similarity_with_edge_cases():
#     input_features, output_features = 256, 256
#     w = Weight(input_features, output_features, init_method='kaiming')
#     model = CosineSimilarity(w, dim=1)

#     # Zero vector test
#     zero_vector = torch.zeros(1, input_features)
#     output_zero = model(zero_vector)
#     assert torch.all(output_zero <= 1), "Cosine similarity with a zero vector should be <= 1"

#     # Negative vector test
#     negative_vector = -torch.ones(1, input_features)
#     output_negative = model(negative_vector)
#     assert torch.all(output_negative >= -1) and torch.all(output_negative <= 1), "Cosine similarity should be between -1 and 1"


# def test_cosine_similarity_for_weird():
#     n_vectors, vec_size = 13, 2048
#     w = Weight(n_vectors, vec_size)
#     model = CosineSimilarity(w, unsqueeze_inputs=[2], unsqueeze_weights=[0, 0, -1, -1], dim=3)
#     ww = w.weight.clone().detach()  # Clone the weight to use as an input

#     inputs = torch.randn(3, 1, vec_size, 5, 1)
#     inputs[0, 0, :, 0, 0] = ww[0]
#     inputs[1, 0, :, 0, 0] = ww[1]
#     inputs[2, 0, :, 0, 0] = ww[2]

#     output = model(inputs)
#     expected = torch.zeros((3, 1, 13, 5, 1))
#     for i in range(3):
#         expected[i, 0, i, 0, 0] = 1.0

#     assert output.shape == expected.shape
#     assert torch.allclose(output, expected, atol=0.1), "Cosine similarity of identical vectors should be close to 1."

# def test_reverse_cosine_similarity_for_weird():

#     '''Set several inputs to be highly similar to the known symbol
#     dictionary. Project forward. Look they're similar! Project backward. We
#     should recover the original high similarity vectors, but the other vectors
#     will likely not be similar to their original vectors.'''

#     n_vectors, vec_size = 13, 2048
#     w = Weight(n_vectors, vec_size)
#     fwd = CosineSimilarity(w, unsqueeze_inputs=[2], unsqueeze_weights=[0, 0, -1, -1], dim=3)
#     bwd = ReverseCosineSimilarity(fwd)

#     ww = w.weight.clone().detach()  # Clone the weight to use as an input

#     inputs = torch.randn(3, 1, vec_size, 5, 1)
#     inputs[0, 0, :, 2, 0] = ww[0]
#     inputs[1, 0, :, 2, 0] = ww[1]
#     inputs[2, 0, :, 2, 0] = ww[2]

#     # Go forward
#     f = fwd(inputs)

#     # Project back
#     b = bwd(f)

#     assert b.shape == inputs.shape


#     expected = torch.zeros((3, 1, 5, 1))
#     for i in range(3):
#         expected[i, 0, 2, 0] = 1.0

#     out = cosine_similarity(b, inputs, dim=2)

#     assert out.shape == expected.shape, f'out: {out.shape}, expected: {expected.shape}'
#     assert torch.allclose(out, expected, atol=0.2), "Cosine similarity of identical vectors should be close to 1."
