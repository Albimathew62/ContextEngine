from contextengine.utils import cosine_similarity


def test_identical_vectors_score_one():
    v = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_orthogonal_vectors_score_zero():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_opposite_vectors_score_minus_one():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(cosine_similarity(a, b) + 1.0) < 1e-6


def test_zero_vector_no_crash():
    result = cosine_similarity([0.0, 0.0], [1.0, 0.0])
    assert abs(result) < 1e-6