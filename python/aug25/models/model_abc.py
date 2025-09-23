from abc import ABC

from ..common_interface import RangeSet


class Model(ABC):
    @staticmethod
    def from_json_string(s: str) -> "Model":
        raise NotImplementedError

    def get_mask(self) -> RangeSet:
        raise NotImplementedError


    def commit(self, token_id: int):
        raise NotImplementedError
