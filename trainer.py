from vector_store import add_example, build_index
from models import TrainingExample

def add_training_example(example: TrainingExample):
    add_example(example)

def reindex():
    build_index()
