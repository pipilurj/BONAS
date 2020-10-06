import random
from opendomain_utils.mutate import all_mutates
from .base_sampler import BaseSampler
from opendomain_utils.transform_genotype import geno_to_archs
class EASampler(BaseSampler):
    def __init__(self, trained_arch_list, population_size=30, tornament_size=10):
        #trained arch list[genostr, acc]
        genotypes = [sample['genotype'] for sample in trained_arch_list]
        ei_scores = [sample['metrics'] for sample in trained_arch_list]
        dataset = geno_to_archs(genotypes, ei_scores)
        self.tornament_size = tornament_size
        self.population = random.sample(dataset, population_size)
        self.candidates = random.sample(self.population, self.tornament_size)

    def sample(self):
        parent = max(self.candidates, key=lambda i: i['metrics'])
        samples = all_mutates(old_arch=parent)
        return samples

    def update_sampler(self, child, ifappend=True, *args, **kwargs):
        # TODO: support update sampler for muliple children in supernet case
        if ifappend:
            if isinstance(child, list):
                self.population.extend(child)
                self.population = self.population[len(child):]
            else:
                self.population.append(child)
                self.population.pop(0)
        print(f"updated sampler, pupulation size:{len(self.population)}")
        self.candidates = random.sample(self.population, self.tornament_size)