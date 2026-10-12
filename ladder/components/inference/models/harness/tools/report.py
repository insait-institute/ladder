import dataclasses


@dataclasses.dataclass(frozen=True)
class Report:
    ok: bool
    text: str

    def __str__(self) -> str:
        return self.text
