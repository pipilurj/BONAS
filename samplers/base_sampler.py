class BaseSampler:
    def sample(self):
        raise NotImplementedError("Such sampler hasn't been implemented")
    def update_sampler(self, *args, **kwargs):
        pass