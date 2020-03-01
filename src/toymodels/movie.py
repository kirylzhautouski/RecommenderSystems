from dataclasses import dataclass


@dataclass(frozen=True)
class Movie:
    name: str
    animated: bool
    is_marvel: bool
    has_super_villain: bool
    passes_bechdel_test: bool

    def match(self, other):
        score = 0
        for attr, value in self.__dict__.items():
            if value == getattr(other, attr):
                score += 1

        return score
