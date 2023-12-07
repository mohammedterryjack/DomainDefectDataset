from argparse import ArgumentParser
from base64 import b64decode
from json import load

from matplotlib.pyplot import show, subplots
from numpy import frombuffer, ndarray

from synthetic_data_generator import (
    fill_domains,
    generate_selected_domain_patterns,
    label_domains_given_defects,
)


def string_to_array(image: str, shape: tuple[int, int]) -> ndarray:
    image_bytes = b64decode(image.encode("utf-8"))
    image_array = frombuffer(image_bytes, dtype=int)
    return image_array.reshape(shape)


def reconstruct_spacetime(
    defects: ndarray, domain_pattern_signatures: list[list[str]]
) -> ndarray:
    synthetic_domains = label_domains_given_defects(image=defects)
    # TODO: fix this - the domains labelled are not identical to the original domains for stochastic examples
    depth, width = synthetic_domains.shape
    return fill_domains(
        n_domains=len(domain_pattern_signatures),
        segmented_image=synthetic_domains,
        background_patterns=generate_selected_domain_patterns(
            width=width, depth=depth, pattern_signatures=domain_pattern_signatures
        ),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    arguments = parser.parse_args()

    with open(arguments.path) as json_file:
        data = load(json_file)
    annotation = string_to_array(
        image=data["annotated_defects"],
        shape=(data["metadata"]["time"], data["metadata"]["lattice_width"]),
    )
    spacetime = reconstruct_spacetime(
        defects=annotation,
        domain_pattern_signatures=[
            domain["pattern_signature"].split("-")
            for domain in data["metadata"]["domains"]
        ],
    )

    fig, axs = subplots(2)
    fig.suptitle(arguments.path)
    axs[0].imshow(spacetime, cmap="gray")
    axs[1].imshow(annotation, cmap="gray")
    show()
