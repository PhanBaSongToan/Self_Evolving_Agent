import numpy as np

from lightweight_router.structural_features import FEATURE_NAMES, StructuralDocumented18


def test_structural_output_shape_and_order():
    transformer = StructuralDocumented18().fit(["Who caused the event?"])
    matrix = transformer.transform(["Who caused the event?", "Compare totals after 2020."])
    assert matrix.shape == (2, 18)
    assert tuple(transformer.get_feature_names_out()) == FEATURE_NAMES


def test_structural_extraction_is_deterministic():
    values = ["Why did Alice compare totals after 2020, and what happened?"]
    first = StructuralDocumented18().fit_transform(values)
    second = StructuralDocumented18().fit_transform(values)
    assert np.array_equal(first, second)
