"""Tool for annotating defects on a spacetime"""

from argparse import ArgumentParser
from eca import OneDimensionalElementaryCellularAutomata
from matplotlib.pyplot import show, figure, imshow, draw, clf
from yaml import safe_load
from numpy import zeros_like, savetxt, loadtxt
from os.path import isfile 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample", type=int, default=1)
    arguments = parser.parse_args()

    with open(f'./dataset/metadata/{arguments.sample}.yaml') as metadata_file:
        metadata = safe_load(metadata_file)

    ca = OneDimensionalElementaryCellularAutomata(
        lattice_width=metadata['width'], 
        initial_configuration= metadata['initial_configuration'] 
    )
    for _ in range(metadata['depth']):
        ca.transition(rule_number=metadata['rule'])

    annotation_path = f'./dataset/annotations/{arguments.sample}.txt'
    if isfile(annotation_path):
        annotation_canvas = loadtxt(annotation_path, dtype=int)
    else:
        annotation_canvas = zeros_like(ca.evolution())

    def toggle_coordinate(event) -> None:
        if event.xdata and event.ydata:
            relative_x = event.xdata/100
            relative_y = event.ydata/50
            x = int(metadata['width'] * relative_x)
            y = int(metadata['depth'] * relative_y)
            annotation_canvas[y][x] = int(not annotation_canvas[y][x])
            clf()
            imshow(ca.evolution(), cmap='gray')
            imshow(annotation_canvas, alpha=0.5)
            draw()
    
    fig = figure()
    fig.canvas.mpl_connect('button_press_event', toggle_coordinate)
    imshow(ca.evolution(), cmap="gray")
    imshow(annotation_canvas, alpha=0.5)
    show()
    draw()

    savetxt(f'./dataset/annotations/{arguments.sample}.txt', annotation_canvas, fmt='%d')
