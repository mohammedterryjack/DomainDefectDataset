from argparse import ArgumentParser
from base64 import b64decode
from json import load

from matplotlib.pyplot import show, subplots
from numpy import frombuffer, ndarray


def string_to_array(image: str, shape: tuple[int, int]) -> ndarray:
    image_bytes = b64decode(image.encode("utf-8"))
    image_array = frombuffer(image_bytes, dtype=int)
    return image_array.reshape(shape)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    arguments = parser.parse_args()

    with open(arguments.path) as json_file:
        data = load(json_file)
    spacetime = string_to_array(
        image=data["spacetime"],
        shape=(data["metadata"]["time"], data["metadata"]["lattice_width"]),
    )
    annotation = string_to_array(
        image=data["annotated_defects"],
        shape=(data["metadata"]["time"], data["metadata"]["lattice_width"]),
    )
    fig, axs = subplots(2)
    fig.suptitle(arguments.path)
    axs[0].imshow(spacetime, cmap="gray")
    axs[1].imshow(annotation, cmap="gray")
    show()
