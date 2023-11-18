"""Tool for viewing a sample in the dataset"""

from argparse import ArgumentParser
from eca import OneDimensionalElementaryCellularAutomata
from matplotlib.pyplot import show, subplots
from yaml import safe_load
from numpy import loadtxt

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
    
    filtered_spacetime = loadtxt(f'./dataset/annotations/{arguments.sample}.txt', dtype=int)

    _, canvas = subplots(2, 1)
    canvas[0].title.set_text(f'Spacetime Diagram (sample {arguments.sample})')
    canvas[0].imshow(ca.evolution(), cmap="gray")
    canvas[1].imshow(filtered_spacetime, cmap="gray")
    show()