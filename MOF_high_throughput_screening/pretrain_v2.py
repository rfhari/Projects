from ocpmodels.preprocessing.atoms_to_graphs_modified import AtomsToGraphs
from ase import io
import pandas as pd
import numpy as np
import yaml 
from schnet_original import *
from collections import OrderedDict
import sys

sys.path.append('../../')

from matdeeplearn import models
import matdeeplearn.process as process
from matdeeplearn.training.training import *

sys.path.remove("../../")

target_index = 0
model_path = "Schnet_OCP.pt"
data_path = "../../data/MOF_k/unit_cell/"
file_name = data_path + "unit_cell_10K_log_train.csv"
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
batch_size = 32
rank = 'cpu' #cpu or cuda
seed = 123
world_size = 1
print("Torch version:",torch.__version__)
print("Is CUDA enabled?",torch.cuda.is_available())

#------------------------------------------------------------------------------
with open('schnet.yml', 'r') as file:
    all_values_config = yaml.safe_load(file)

model_parameters = all_values_config["model"]
training_parameters = all_values_config["Training"]
job_parameters = all_values_config["job"]
processing_parameters = all_values_config["model"]

saved = torch.load(model_path, map_location=torch.device("cpu"))

model_original = SchNetWrap(**processing_parameters)
new_state_dict = OrderedDict()
for k, v in saved["state_dict"].items():
    name = k[14:] 
    new_state_dict[name] = v
model_original.load_state_dict(new_state_dict)
model_original.to(rank)

#------------------------------------------------------------------------------

ase_atoms_list = []
labels_atoms_list = []
structure_ids = []
raw_data = pd.read_csv(file_name)
atoms_data = raw_data.iloc[:, 0]
labels_collections_all = np.mean(raw_data.iloc[:, 1:], axis=1)
# print(type(labels_collections_all))

for idx, cif_name in enumerate(atoms_data):
    structure_ids.append(cif_name)
    ase_atoms_list.append(io.read(data_path+"/"+cif_name+".cif"))
    # print(ase_atoms_list)
    labels_atoms_list.append(labels_collections_all[idx])

a2g = AtomsToGraphs(
    max_neigh=12,
    radius=6,
    r_energy=True, #use thermal conductivity data instead of energy 
    r_forces=False,
    r_distances=False,
    r_edges=True,
    r_fixed=True,
)

data_objects = a2g.convert_all(ase_atoms_list, structure_id=structure_ids, thermal_conductivity=labels_atoms_list, disable_tqdm=True)

(train_loader, val_loader, test_loader, train_sampler, train_dataset,  _,  _,) = loader_setup(  train_ratio,
                                                                                                val_ratio,
                                                                                                test_ratio,
                                                                                                batch_size,
                                                                                                data_objects,
                                                                                                rank,
                                                                                                seed,
                                                                                                world_size,
                                                                                            )
# for batch in train_loader:
#     print(batch.y)

optimizer = getattr(torch.optim, training_parameters["optimizer"])(
        model_original.parameters(),
        lr=training_parameters["lr"],
        **training_parameters["optimizer_args"]
    )

scheduler = getattr(torch.optim.lr_scheduler, training_parameters["scheduler"])(
        optimizer, **training_parameters["scheduler_args"]
    )

print("Start training")

model = trainer(
        rank,
        world_size,
        model_original,
        optimizer,
        scheduler,
        training_parameters["loss"],
        train_loader,
        val_loader,
        train_sampler,
        training_parameters["epochs"],
        training_parameters["verbosity"],
        "my_model_temp.pth",
    )

if rank in (0, "cpu", "cuda"):
    train_error = val_error = test_error = float("NaN")

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_parameters["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    ##Get train error in eval mode
    train_error, train_out = evaluate(
        train_loader, model, training_parameters["loss"], rank, out=True
    )

    print("Train Error: {:.5f}".format(train_error))

    ##Get val error
    if val_loader != None:
        val_error, val_out = evaluate(
            val_loader, model, training_parameters["loss"], rank, out=True
        )
        print("Val Error: {:.5f}".format(val_error))

    ##Get test error
    if test_loader != None:
        test_error, test_out = evaluate(
            test_loader, model, training_parameters["loss"], rank, out=True
        )
        print("Test Error: {:.5f}".format(test_error))

    ##Save model
    if job_parameters["save_model"] == "True":

        if rank not in ("cpu", "cuda"):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "full_model": model,
                },
                job_parameters["model_path"],
            )
        else:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "full_model": model,
                },
                job_parameters["model_path"],
            )

    if job_parameters["write_output"] == "True":
        write_results(
            train_out, str(job_parameters["job_name"]) + "_train_outputs.csv"
        )
        if val_loader != None:
            write_results(
                val_out, str(job_parameters["job_name"]) + "_val_outputs.csv"
            )
        if test_loader != None:
            write_results(
                test_out, str(job_parameters["job_name"]) + "_test_outputs.csv"
            )

    if rank not in ("cpu", "cuda"):
        dist.destroy_process_group()

    ##Write out model performance to file
    error_values = np.array((train_error.cpu(), val_error.cpu(), test_error.cpu()))
    if job_parameters.get("write_error") == "True":
        np.savetxt(
            job_parameters["job_name"] + "_errorvalues.csv",
            error_values[np.newaxis, ...],
            delimiter=",",
        )