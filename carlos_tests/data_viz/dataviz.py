import pandas as pd
import sqlite3
from graphnet.data.constants import FEATURES, TRUTH
from typing import Any, Dict, List, Optional
import torch
from graphnet.models.graphs import KNNGraph
import os
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.data.datamodule import GraphNeTDataModulecustom
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.training.labels import JointLabel
from datetime import datetime
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt


def load_list_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path, dtype={'event_no': int})
    event_list = df['event_no'].tolist()
    return event_list


# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86
truth.append("oneweight")

db_path = "/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/my_numu_database_part_1 (1).db"

NumuValidation = '/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnumu_validation_selection.csv'
NumuTraining = '/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnumu_training_selection.csv'
NueValidation = '/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnue_validation_selection.csv'
NueTraining = '/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnue_training_selection.csv'

NuMu_Training_Selections = load_list_from_csv(NumuTraining)
NuMu_Validation_Selections = load_list_from_csv(NumuValidation)
NuE_Training_Selections = load_list_from_csv(NueTraining)
NuE_Validation_Selections = load_list_from_csv(NueValidation)

if __name__ == '__main__':
    # Configuration
    config: Dict[str, Any] = {
        "path": db_path,
        "pulsemap": "SRTInIcePulses",  # Name of pulsemap to use
        "batch_size": 128,
        "num_workers": 24,
        "target": "direction",  # Name of feature to use as regression target
        "early_stopping_patience": 2,
        "fit": {
            "gpus": list(range(torch.cuda.device_count())),
            "max_epochs": 1,
        },
    }

    graph_definition = KNNGraph(detector=IceCube86(),
                                node_definition=NodesAsPulses(),
                                nb_nearest_neighbours=8,
                                input_feature_names=features, )

    archive = os.path.join(
        '/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/VertexReco/Vertex/LargeTC0.01_LRNEW/Test3',
        "ResultsFolder")

    data_module = GraphNeTDataModulecustom(
        dataset_reference=SQLiteDataset,
        dataset_args={
            "truth_table": "truth",
            "pulsemaps": config["pulsemap"],
            "truth": truth,
            "features": features,
            "path": [
                "/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/my_numu_database_part_1 (1).db",
                "/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/my_nue_database_part_1 (1).db"
            ],
            "graph_definition": graph_definition},
        train_dataloader_kwargs={"batch_size": config["batch_size"],
                                 "num_workers": config["num_workers"],
                                 },
        train_selections=[NuMu_Training_Selections, NuE_Training_Selections
                          # , NuGen_Training_Selections[:10000]
                          ],
        val_selections=[NuMu_Validation_Selections, NuE_Validation_Selections],
        test_selection=[None, None],
        labels={
            "joint_labels": JointLabel(
                azimuth_key="azimuth", zenith_key="zenith",
                position_keys=("position_x", "position_y", "position_z"),
                key="joint_labels"
            )
        },
        train_val_split=[0.2, 0.8],
    )

    training_dataloader = data_module.train_dataloader
    validation_dataloader = data_module.val_dataloader

    # sampledata.x is a [n,7] list with ['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area']
    # for i in range(100):
    #     sample_data = training_dataloader.dataset.datasets[0][i]
    #
    #     # i = 88
    #     # while sample_data.n_pulses < 128:
    #     #     i += 1
    #     #     sample_data = training_dataloader.dataset.datasets[0][i]
    #
    #     # Let's find the first and last times
    #     t_min = sample_data.x[4].min()
    #     t_max = sample_data.x[4].max()
    #
    #     # We'll now plot the event in 3D. Time of arrival with be color and charge will be size
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(111, projection='3d')
    #
    #     # Set x,y,z range manually to -1,1
    #     ax.set_xlim([-1, 1])
    #     ax.set_ylim([-1, 1])
    #     ax.set_zlim([-1, 1])
    #
    #     # Normalize the charge for size
    #     charge = sample_data.x[5]
    #     norm_charge = (charge - charge.min()) / (charge.max() - charge.min()) * 90 + 10  # Scale to [10, 100]
    #
    #     # Normalize the time for color
    #     time = sample_data.x[4]
    #     norm_time = (time - t_min) / (t_max - t_min)  # Scale to [0, 1]
    #
    #     scatter = ax.scatter(sample_data.x[0], sample_data.x[1], sample_data.x[2],
    #                          c=norm_time, s=norm_charge, cmap='viridis', alpha=0.7)
    #
    #     ax.set_xlabel('X Position')
    #     ax.set_ylabel('Y Position')
    #     ax.set_zlabel('Z Position')
    #     ax.set_title('3D Event Visualization with Time and Charge')
    #
    #     plt.colorbar(scatter, label='Normalized Time of Arrival')
    #     # Save as png
    #     plt.savefig(f'plots/event_visualization_{i}.png', dpi=300, bbox_inches='tight')
    #     plt.close()

    # Range over all events and store their n_pulses, then plot a histogram
    pulses = []
    for sample_data in tqdm(training_dataloader.dataset.datasets[0]):
        pulses.append(sample_data.n_pulses)

    # Save pulses as a numpy array
    np.save('pulses.npy', pulses)

    plt.figure(figsize=(10, 6))
    plt.hist(np.log10(pulses), bins=50, color='blue', alpha=0.7,density=True)
    plt.title('Number of Pulses per Event')
    plt.xlabel(r'$\log_{10}$ Number of Pulses')
    plt.yscale('log')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig('plots/pulses_per_event_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
