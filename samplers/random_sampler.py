import random
from data_generators.dynamic_generate import generate_archs
from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self, generate_num=100):
        self.generate_num = generate_num

    def sample(self):
        samples = generate_archs(generate_num=self.generate_num)
        return samples
