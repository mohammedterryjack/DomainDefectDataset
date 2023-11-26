from itertools import combinations, permutations
from math import dist
from random import randint, sample, shuffle
from typing import Generator, Optional

from matplotlib.pyplot import imshow, show
from numpy import ndarray, ones_like, where, zeros


def generate_domain_pattern(
    width: int,
    depth: int,
    patterns: list[str],
    phase: int,
    pattern_sequence: Optional[list[int]] = None,
) -> tuple[list[str], list[list[int]]]:
    max_n_patterns = len(patterns)
    if pattern_sequence:
        assert (
            len(pattern_sequence) == phase
        ), f"Insufficient patterns for phase of {phase}"
        assert all(
            0 <= pattern_index <= max_n_patterns for pattern_index in pattern_sequence
        ), "Invalid values in pattern_sequence. Each value should be an index of the pattern_sequence"
    else:
        pattern_sequence = sample(range(max_n_patterns), phase)

    pattern_signature = [patterns[pattern_index] for pattern_index in pattern_sequence]
    rows = []
    for _ in range(depth):
        for pattern in pattern_signature:
            rows.append(list(map(int, (pattern * width)[:width])))
    return pattern_signature, rows[:depth]


def generate_domain_patterns(
    max_phase: int, width: int, depth: int
) -> list[tuple[list[str], list[list[int]]]]:
    patterns = ["0", "1", "01", "10", "010", "101", "011", "110"]
    max_n_patterns = len(patterns)
    domain_patterns = []
    for phase in range(1, max_phase + 1):
        for pattern_sequence_head in combinations(range(max_n_patterns), phase):
            for pattern_sequence in permutations(pattern_sequence_head):
                domain_patterns.append(
                    generate_domain_pattern(
                        width=width,
                        depth=depth,
                        phase=phase,
                        patterns=patterns,
                        pattern_sequence=pattern_sequence,
                    )
                )
    return domain_patterns


def random_domains(
    n: int, width: int, depth: int, max_phase: int
) -> Generator[list[list[int]], None, None]:
    domains = generate_domain_patterns(width=width, depth=depth, max_phase=max_phase)
    shuffle(domains)
    for domain_signature, domain_pattern in domains[:n]:
        print(f"Domain Pattern Signature: {domain_signature}")
        yield domain_pattern


def fill_domains(n_domains: int, labelled_image: ndarray) -> ndarray:
    width, height = labelled_image.shape
    domains = list(random_domains(n=n_domains, width=width, depth=height, max_phase=3))
    filled_image = ones_like(labelled_image) * -1
    for domain_label in range(n_domains + 1):
        for x, y in zip(*where(labelled_image == domain_label)):
            filled_image[x][y] = domains[domain_label][x][y]
    return filled_image


def closest(
    target_coord: tuple[int, int], source_coords: list[tuple[int, int]]
) -> float:
    distances = [dist(coord, target_coord) for coord in source_coords]
    return distances.index(min(distances))


def random_coordinates(n: int, width: int, depth: int) -> list[tuple[int, int]]:
    return [(randint(0, width), randint(0, depth)) for _ in range(n)]


def randomly_segmented_image(width: int, depth: int, n_domains: int) -> ndarray:
    seeds = random_coordinates(n=n_domains, width=width, depth=depth)
    labelled_image = zeros(shape=(width, depth))
    for x in range(width):
        for y in range(depth):
            labelled_image[x][y] = closest(target_coord=(x, y), source_coords=seeds)
    return labelled_image


def generate_sample(width: int, depth: int, n_domains: int) -> ndarray:
    return fill_domains(
        n_domains=n_domains,
        labelled_image=randomly_segmented_image(
            width=width, depth=depth, n_domains=n_domains
        ),
    )


image = generate_sample(width=100, depth=100, n_domains=3)
imshow(image, cmap="gray")
show()
