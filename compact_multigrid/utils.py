from compact_multigrid.typing.field import Location


def tuples2locs(tuples: list[tuple[int, int]]) -> list[Location]:
    locs: list[Location] = [Location(*item) for item in tuples]
    return locs
