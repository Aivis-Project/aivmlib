import enum


class StrEnum(str, enum.Enum):
    """
    Enum where members are also (and must be) strings (backported from Python 3.11).
    """

    def __new__(cls, *values: str) -> 'StrEnum':
        "values must already be of type `str`"
        if len(values) > 3:
            raise TypeError(f'too many arguments for str(): {values!r}')
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):  # type: ignore
                raise TypeError(f'{values[0]!r} is not a string')
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):  # type: ignore
                raise TypeError(f'encoding must be a string, not {values[1]!r}')
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):  # type: ignore
                raise TypeError(f'errors must be a string, not {values[2]!r}')
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()
