from injector import Module, Binder
from domain.irepository.sample_repository import ISampleRepository
from repository.sample_repository import SampleRepository

class SampleRegistory(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(ISampleRepository, SampleRepository)
