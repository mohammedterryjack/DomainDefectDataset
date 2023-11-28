from argparse import ArgumentParser
from itertools import combinations, permutations
from math import dist
from random import randint, shuffle

from matplotlib.pyplot import imshow, show
from numpy import ndarray, ones_like, where, zeros


def generate_domain_pattern_from_pattern_signature(
    width: int,
    depth: int,
    pattern_signature: list[str],
) -> list[list[int]]:
    rows = []
    for _ in range(depth):
        for pattern in pattern_signature:
            rows.append(list(map(int, (pattern * width)[:width])))
    return rows[:depth]


def generate_selected_domain_patterns(
    width: int, depth: int, pattern_signatures: list[list[str]]
) -> list[list[int]]:
    domain_patterns = []
    for pattern_signature in pattern_signatures:
        domain_patterns.append(
            generate_domain_pattern_from_pattern_signature(
                width=width,
                depth=depth,
                pattern_signature=pattern_signature,
            )
        )
    return domain_patterns


def generate_every_domain_signature(max_phase: int) -> list[list[str]]:
    patterns = ["0", "1", "01", "10", "010", "101", "011", "110"]
    max_n_patterns = len(patterns)
    domain_signatures = []
    for phase in range(1, max_phase + 1):
        for pattern_sequence_head in combinations(range(max_n_patterns), phase):
            for pattern_sequence in permutations(pattern_sequence_head):
                domain_signatures.append(
                    [patterns[pattern_index] for pattern_index in pattern_sequence]
                )
    return domain_signatures


def random_domains(
    n: int, width: int, depth: int, max_phase: int
) -> tuple[list[str], list[list[int]]]:
    domain_signatures = generate_every_domain_signature(max_phase=max_phase)
    shuffle(domain_signatures)
    return domain_signatures[:n], generate_selected_domain_patterns(
        width=width, depth=depth, pattern_signatures=domain_signatures[:n]
    )


def fill_domains(
    n_domains: int, segmented_image: ndarray, background_patterns: list[list[list[int]]]
) -> ndarray:
    filled_image = ones_like(segmented_image) * -1
    for domain_label in range(n_domains):
        for x, y in zip(*where(segmented_image == domain_label)):
            background_pattern = background_patterns[domain_label]
            filled_image[x][y] = background_pattern[y][x]
    return filled_image


def closest_coordinates(
    target_coordinate: tuple[int, int], source_coordinates: list[tuple[int, int]]
) -> float:
    distances = [dist(coord, target_coordinate) for coord in source_coordinates]
    return distances.index(min(distances))


def random_coordinates(n: int, width: int, depth: int) -> list[tuple[int, int]]:
    return [(randint(0, width - 1), randint(0, depth - 1)) for _ in range(n)]


def segment_image_by_distance_from_seed(
    width: int,
    depth: int,
    seed_coordinates: list[tuple[int, int]],
) -> ndarray:
    labelled_image = zeros(shape=(width, depth))
    for x in range(width):
        for y in range(depth):
            labelled_image[x][y] = closest_coordinates(
                target_coordinate=(x, y), source_coordinates=seed_coordinates
            )
    return labelled_image


def generate_sample(
    width: int,
    depth: int,
    n_domains: int,
    domain_seed_coordinates: list[tuple[int, int]],
    domain_pattern_signatures: list[str],
) -> ndarray:
    domain_patterns = generate_selected_domain_patterns(
        width=width, depth=depth, pattern_signatures=domain_pattern_signatures
    )
    segmented_image = segment_image_by_distance_from_seed(
        width=width, depth=depth, seed_coordinates=domain_seed_coordinates
    )
    return fill_domains(
        n_domains=n_domains,
        segmented_image=segmented_image,
        background_patterns=domain_patterns,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--time", type=int, default=200)
    parser.add_argument("--max_phase", type=int, default=None)
    parser.add_argument("--n_domains", type=int, default=None)
    parser.add_argument(
        "--domain_centre", type=int, nargs="+", action="append", default=None
    )
    parser.add_argument(
        "--domain_pattern", type=str, nargs="+", action="append", default=None
    )
    arguments = parser.parse_args()

    width = arguments.width
    depth = arguments.time
    max_phase_domain_pattern = arguments.max_phase
    domain_seed_coordinates = arguments.domain_centre
    domain_pattern_signatures = arguments.domain_pattern
    n_domains = arguments.n_domains

    if n_domains is None:
        if domain_seed_coordinates:
            n_domains = len(domain_seed_coordinates)
        elif domain_pattern_signatures:
            n_domains = len(domain_pattern_signatures)
        else:
            n_domains = randint(2, 5)

    assert depth > 1, "depth must be a positive integer"
    assert width > 1, "width must be a positive integer"
    assert n_domains > 1, "number of domains must be more than one"

    if domain_seed_coordinates is None:
        domain_seed_coordinates = random_coordinates(
            n=n_domains, width=width, depth=depth
        )

    assert (
        len(domain_seed_coordinates) == n_domains
    ), f"number of domain seeds should match number of domains specified ({n_domains})"
    assert all(
        0 <= x <= width and 0 <= y <= depth for x, y in domain_seed_coordinates
    ), f"all coordinates must be within the bounds of the image (width={width} depth={depth})"

    if max_phase_domain_pattern is None:
        if domain_pattern_signatures:
            max_phase_domain_pattern = max(map(len, domain_pattern_signatures))
        else:
            max_phase_domain_pattern = randint(2, 5)

    assert max_phase_domain_pattern > 1

    if domain_pattern_signatures is None:
        domain_pattern_signatures, domain_patterns = random_domains(
            n=n_domains, width=width, depth=depth, max_phase=max_phase_domain_pattern
        )

    assert len(domain_pattern_signatures) == len(
        set(map("-".join, domain_pattern_signatures))
    ), f"each domain pattern signature should be unique: {domain_pattern_signatures}"
    assert (
        len(domain_pattern_signatures) == n_domains
    ), f"number of pattern signatures should match number of domains requested ({n_domains})"
    assert all(
        1 <= len(domain_pattern_signature) <= max_phase_domain_pattern
        for domain_pattern_signature in domain_pattern_signatures
    )

    print(
        f"width: {width}\ndepth: {depth}\ndomain seed coordinates: {domain_seed_coordinates}\ndomain pattern signatures: {domain_pattern_signatures}"
    )

    image = generate_sample(
        width=width,
        depth=depth,
        n_domains=n_domains,
        domain_seed_coordinates=domain_seed_coordinates,
        domain_pattern_signatures=domain_pattern_signatures,
    )
    imshow(image, cmap="gray")
    show()
