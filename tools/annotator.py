"""Tool for annotating defects on a spacetime"""

from argparse import ArgumentParser
from os.path import isfile

from eca import OneDimensionalElementaryCellularAutomata
from matplotlib.pyplot import clf, draw, figure, imshow, show
from numpy import loadtxt, savetxt, zeros_like
from yaml import safe_load

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample", type=int, default=1)
    arguments = parser.parse_args()

    with open(f"./dataset/metadata/{arguments.sample}.yaml") as metadata_file:
        metadata = safe_load(metadata_file)

    ca = OneDimensionalElementaryCellularAutomata(
        lattice_width=metadata["width"],
        initial_configuration=metadata["initial_configuration"],
    )
    for _ in range(metadata["depth"]):
        ca.transition(rule_number=metadata["rule"])

    annotation_path = f"./dataset/annotations/{arguments.sample}.txt"
    if isfile(annotation_path):
        annotation_canvas = loadtxt(annotation_path, dtype=int)
    else:
        annotation_canvas = zeros_like(ca.evolution())
    pen_down = False
    prev_coords = []

    def toggle_pen(event) -> None:
        global pen_down
        global prev_coords
        pen_down = not pen_down
        prev_coords = []

    def toggle_coordinate(event) -> None:
        global prev_coords
        global pen_down
        if event.xdata and event.ydata and pen_down:
            x = int(event.xdata)
            y = int(event.ydata)
            if (x, y) not in prev_coords:
                prev_coords.append((x, y))
                annotation_canvas[y][x] = int(not annotation_canvas[y][x])
                clf()
                imshow(ca.evolution(), cmap="gray")
                imshow(annotation_canvas, alpha=0.5)
                draw()

    fig = figure()
    fig.canvas.mpl_connect("button_press_event", toggle_pen)
    fig.canvas.mpl_connect("motion_notify_event", toggle_coordinate)
    imshow(ca.evolution(), cmap="gray")
    imshow(annotation_canvas, alpha=0.5)
    show()
    draw()

    savetxt(
        f"./dataset/annotations/{arguments.sample}.txt", annotation_canvas, fmt="%d"
    )
