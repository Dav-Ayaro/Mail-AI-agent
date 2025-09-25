import pytest
from vector_store import add_example, build_index, find_best_match
from models import TrainingExample

def test_vector_store_basic():
    ex = TrainingExample(email="I need help with my transfer", reply="Please share your transaction ID")
    add_example(ex)
    build_index()
    match, confidence = find_best_match("transfer problem")
    assert match is not None
    assert confidence >= 0
