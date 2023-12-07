from argparse import ArgumentParser, BooleanOptionalAction
from base64 import b64encode
from itertools import combinations, permutations
from json import dump
from math import dist
from random import randint, shuffle

from cv2 import connectedComponents
from matplotlib.pyplot import show, subplots
from numpy import array2string, logical_not, ndarray, ones_like, where, zeros
from numpy.random import randint


def generate_random_signature(width: int, phase: int) -> list[str]:
    space_pattern = randint(2, size=(phase, width))
    signature = [
        array2string(space_pattern[row_index], separator="")
        .replace("\n", "")
        .replace(" ", "")
        .strip("[")
        .strip("]")
        for row_index in range(len(space_pattern))
    ]
    return signature


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


def generate_every_domain_signature(min_phase: int, max_phase: int) -> list[list[str]]:
    patterns = ["0", "1", "01", "10", "010", "101", "011", "110"]
    max_n_patterns = len(patterns)
    domain_signatures = []
    for phase in range(min_phase, max_phase + 1):
        for pattern_sequence_head in combinations(range(max_n_patterns), phase):
            for pattern_sequence in permutations(pattern_sequence_head):
                domain_signatures.append(
                    [patterns[pattern_index] for pattern_index in pattern_sequence]
                )
    return domain_signatures


def regular_domains(n: int, min_phase: int, max_phase: int) -> list[list[str]]:
    domain_signatures = generate_every_domain_signature(
        min_phase=min_phase, max_phase=max_phase
    )
    shuffle(domain_signatures)
    return domain_signatures[:n]


def irregular_domains(n: int, min_phase: int, max_phase: int) -> list[list[str]]:
    return [
        generate_random_signature(width=width, phase=randint(min_phase, max_phase))
        for _ in range(n)
    ]


def fill_domains(
    n_domains: int, segmented_image: ndarray, background_patterns: list[list[list[int]]]
) -> ndarray:
    filled_image = ones_like(segmented_image) * -1
    for domain_label in range(n_domains):
        for x, y in zip(*where(segmented_image == domain_label)):
            background_pattern = background_patterns[domain_label]
            filled_image[x][y] = background_pattern[x][y]
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
    labelled_image = zeros(shape=(depth, width))
    for x in range(depth):
        for y in range(width):
            labelled_image[x][y] = closest_coordinates(
                target_coordinate=(x, y), source_coordinates=seed_coordinates
            )
    return labelled_image


def find_domain_boundaries_using_neighbours(segmented_image: ndarray) -> ndarray:
    width, depth = segmented_image.shape
    contours = zeros(shape=(width, depth))
    for x in range(width):
        for y in range(depth):
            centre = segmented_image[x][y]
            right = segmented_image[min(x + 1, width - 1)][y]
            left = segmented_image[max(x - 1, 0)][y]
            up = segmented_image[x][min(y + 1, depth - 1)]
            down = segmented_image[x][max(y - 1, 0)]
            if centre == right == left == up == down:
                continue
            contours[x][y] = 1
    return contours


def add_random_walk_defect(image: ndarray) -> None:
    def is_within_bounds(x: int, min_x: int, max_x: int) -> bool:
        return min_x <= x < max_x

    height, width = image.shape
    x = randint(0, width)
    for y in range(height):
        delta = randint(-1, 1)
        x += delta
        if not is_within_bounds(x=x, min_x=0, max_x=width):
            break
        if image[y][x] == 1:
            break
        image[y][x] = 1


def add_random_walk_defects(image: ndarray, n: int) -> None:
    for _ in range(n):
        add_random_walk_defect(image=image)


def label_domains_given_defects(image: ndarray) -> None:
    _, label_ids = connectedComponents(
        image=logical_not(image).astype("int8"),
        connectivity=4,
    )
    return label_ids


def array_to_string(image: ndarray) -> str:
    return b64encode(image).decode("utf-8")


def generate_sample(
    width: int,
    depth: int,
    n_domains: int,
    domain_seed_coordinates: list[tuple[int, int]],
    domain_pattern_signatures: list[list[str]],
) -> tuple[ndarray, ndarray, ndarray]:
    """Generate a synthetic spacetime-like pattern and annotations

    Args:
        width (int): the width of the synthetic spacetime-like pattern
        depth (int): the depth of the synthetic spacetime-like pattern
        n_domains (int): the number of domains the synthetic image should have
        domain_seed_coordinates (list[tuple[int,int]]): the coordinate of the centre of each domain
        domain_pattern_signatures (list[str]): the patterns for each domain

    Returns:
        ndarray: synthetic spacetime-like pattern
        ndarray: an annotation of the domains
        ndarray: an annotation of the domain defects
    """
    synthetic_domains = segment_image_by_distance_from_seed(
        width=width, depth=depth, seed_coordinates=domain_seed_coordinates
    )
    synthetic_domain_defects = find_domain_boundaries_using_neighbours(
        segmented_image=synthetic_domains
    )
    synthetic_spacetime = fill_domains(
        n_domains=n_domains,
        segmented_image=synthetic_domains,
        background_patterns=generate_selected_domain_patterns(
            width=width, depth=depth, pattern_signatures=domain_pattern_signatures
        ),
    )
    return (
        synthetic_spacetime,
        synthetic_domains,
        synthetic_domain_defects,
    )


def generate_stochastic_sample(
    width: int, depth: int, n_defects: int, max_phase: int, min_phase: int
) -> tuple[ndarray, ndarray, ndarray, list[str]]:
    """Generate a synthetic spacetime-like pattern and annotations with stochastic defects and complex domains

    Args:
        width (int): the width of the synthetic spacetime-like pattern
        depth (int): the depth of the synthetic spacetime-like pattern
        n_defects (int): the number of stochastic defects the synthetic image should have

    Returns:
        ndarray: synthetic spacetime-like pattern
        ndarray: an annotation of the domains
        ndarray: an annotation of the domain defects
        list[str]: domain pattern signatures
    """

    synthetic_domain_defects = zeros((depth, width))
    add_random_walk_defects(image=synthetic_domain_defects, n=n_defects)
    synthetic_domains = label_domains_given_defects(image=synthetic_domain_defects)
    n_domains = synthetic_domains.max() + 1
    domain_pattern_signatures = irregular_domains(
        n=n_domains,
        min_phase=min_phase,
        max_phase=max_phase,
    )
    synthetic_spacetime = fill_domains(
        n_domains=n_domains,
        segmented_image=synthetic_domains,
        background_patterns=generate_selected_domain_patterns(
            width=width, depth=depth, pattern_signatures=domain_pattern_signatures
        ),
    )
    return (
        synthetic_spacetime,
        synthetic_domains,
        synthetic_domain_defects,
        domain_pattern_signatures,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--space", type=int, default=200)
    parser.add_argument("--time", type=int, default=100)
    parser.add_argument("--min_phase", type=int, default=None)
    parser.add_argument("--max_phase", type=int, default=None)
    parser.add_argument("--n_domains", type=int, default=None)
    parser.add_argument(
        "--domain_centre", type=int, nargs="+", action="append", default=None
    )
    parser.add_argument(
        "--domain_pattern", type=str, nargs="+", action="append", default=None
    )
    parser.add_argument("--display", action=BooleanOptionalAction, default=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--stochastic_defects", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="dataset")
    arguments = parser.parse_args()

    n_samples = arguments.samples
    width = arguments.space
    depth = arguments.time
    min_phase_domain_pattern = arguments.min_phase
    max_phase_domain_pattern = arguments.max_phase
    domain_seed_coordinates = arguments.domain_centre
    domain_pattern_signatures = arguments.domain_pattern
    n_domains = arguments.n_domains
    n_stochastic_defects = arguments.stochastic_defects

    assert n_samples > 0, "n_samples must be a positive integer"
    for sample_number in range(n_samples):
        if arguments.n_domains is None:
            if domain_seed_coordinates:
                n_domains = len(domain_seed_coordinates)
            elif domain_pattern_signatures:
                n_domains = len(domain_pattern_signatures)
            else:
                n_domains = randint(2, 5)

        assert depth > 1, "depth must be a positive integer"
        assert width > 1, "width must be a positive integer"
        assert n_domains > 1, "number of domains must be more than one"

        if arguments.domain_centre is None:
            domain_seed_coordinates = random_coordinates(
                n=n_domains, width=width, depth=depth
            )

        assert (
            len(domain_seed_coordinates) == n_domains
        ), f"number of domain seeds should match number of domains specified ({n_domains})"
        assert all(
            0 <= x <= width and 0 <= y <= depth for x, y in domain_seed_coordinates
        ), f"all coordinates must be within the bounds of the image (width={width} depth={depth})"

        if min_phase_domain_pattern is None:
            if domain_pattern_signatures:
                min_phase_domain_pattern = min(map(len, domain_pattern_signatures))
            elif n_stochastic_defects:
                min_phase_domain_pattern = 5
            else:
                min_phase_domain_pattern = 2
        if max_phase_domain_pattern is None:
            if domain_pattern_signatures:
                max_phase_domain_pattern = max(map(len, domain_pattern_signatures))
            elif n_stochastic_defects:
                max_phase_domain_pattern = 10
            else:
                max_phase_domain_pattern = 5

        assert max_phase_domain_pattern > 0

        if arguments.domain_pattern is None:
            domain_pattern_signatures = regular_domains(
                n=n_domains,
                min_phase=min_phase_domain_pattern,
                max_phase=max_phase_domain_pattern,
            )

        assert len(domain_pattern_signatures) == len(
            set(map("-".join, domain_pattern_signatures))
        ), f"each domain pattern signature should be unique: {domain_pattern_signatures}"
        assert (
            len(domain_pattern_signatures) == n_domains
        ), f"number of pattern signatures should match number of domains requested ({n_domains})"
        assert all(
            min_phase_domain_pattern
            <= len(domain_pattern_signature)
            <= max_phase_domain_pattern
            for domain_pattern_signature in domain_pattern_signatures
        )

        if n_stochastic_defects:
            (
                spacetime,
                domains,
                defects,
                domain_pattern_signatures,
            ) = generate_stochastic_sample(
                width=width,
                depth=depth,
                n_defects=n_stochastic_defects,
                max_phase=max_phase_domain_pattern,
                min_phase=min_phase_domain_pattern,
            )

        else:
            spacetime, domains, defects = generate_sample(
                width=width,
                depth=depth,
                n_domains=n_domains,
                domain_seed_coordinates=domain_seed_coordinates,
                domain_pattern_signatures=domain_pattern_signatures,
            )

        if arguments.display:
            fig, axs = subplots(3)
            fig.suptitle("Synthetic Sample")
            axs[0].imshow(spacetime, cmap="gray")
            axs[1].imshow(domains, cmap="gray")
            axs[2].imshow(defects, cmap="gray")
            show()

        with open(
            f"{arguments.save_path}/sample_{sample_number}.json", "w"
        ) as save_file:
            data = dict(
                metadata=dict(
                    lattice_width=width,
                    time=depth,
                    domains=list(
                        map(
                            lambda centre, signature: dict(
                                centre=dict(x=centre[0], y=centre[1]),
                                pattern_signature="-".join(signature),
                            ),
                            domain_seed_coordinates,
                            domain_pattern_signatures,
                        )
                    ),
                    n_stochastic_defects=n_stochastic_defects,
                ),
                annotated_defects=array_to_string(defects),
            )
            dump(data, save_file, indent=3)
