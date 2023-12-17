from argparse import ArgumentParser
from base64 import b64decode
from json import load

from matplotlib.pyplot import show, subplots
from numpy import array, frombuffer, ndarray

from synthetic_data_generator import fill_domains, generate_selected_domain_patterns


def string_to_array(image: str, shape: tuple[int, int]) -> ndarray:
    image_bytes = b64decode(image.encode("utf-8"))
    image_array = frombuffer(image_bytes, dtype=int)
    return image_array.reshape(shape).astype(bool).astype(int)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    arguments = parser.parse_args()

    with open(arguments.path) as json_file:
        data = load(json_file)
    domain_pattern_signatures = [
        domain["pattern_signature"].split("-") for domain in data["metadata"]["domains"]
    ]
    defects = string_to_array(
        image=data["annotated_defects"],
        shape=(data["metadata"]["time"], data["metadata"]["lattice_width"]),
    )
    domains = array(data["domain_regions"], dtype=int)
    spacetime = fill_domains(
        n_domains=len(domain_pattern_signatures),
        segmented_image=domains,
        background_patterns=generate_selected_domain_patterns(
            width=data["metadata"]["lattice_width"],
            depth=data["metadata"]["time"],
            pattern_signatures=domain_pattern_signatures,
        ),
    )

    fig, axs = subplots(3)
    fig.suptitle(arguments.path)
    axs[0].imshow(spacetime, cmap="gray")
    axs[1].imshow(domains)
    axs[2].imshow(defects, cmap="gray")
    show()
