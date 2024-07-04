import os 
import argparse
from os.path import join
import torch
import pickle
from qm9.models import get_model
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.sampling import sample
from qm9.property_prediction.main_qm9_prop import test
from qm9.property_prediction import main_qm9_prop
from qm9.sampling import sample_chain, sample, sample_sweep_conditional, sample_sweep_guidance
import qm9.visualizer as vis
from qm9.property_prediction import prop_utils
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_classifier(dir_path='', device='cpu'):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = main_qm9_prop.get_model(args_classifier)
    classifier_state_dict = torch.load(join(dir_path, 'best_checkpoint.npy'), map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_state_dict)

    return classifier


def get_args_gen(dir_path):
    # breakpoint()
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_gen = pickle.load(f)
    assert args_gen.dataset == 'qm9_second_half'

    # Add missing args!
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'
    return args_gen


def get_generator(dir_path, dataloaders, device, args_gen, property_norms, ckpt=None):
    dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)
    model, nodes_dist, prop_dist = get_model(args_gen, device, dataset_info, dataloaders['train'])
    fn = 'generative_model_ema.npy' if args_gen.ema_decay > 0 else 'generative_model.npy'
    if ckpt:
        fn =f"generative_model_{ckpt}.npy" if args_gen.ema_decay > 0 else f"generative_model_{ckpt}.npy"
    model_state_dict = torch.load(join(dir_path, fn), map_location='cpu')
    model.load_state_dict(model_state_dict)

    # The following function be computes the normalization parameters using the 'valid' partition

    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    return model.to(device), nodes_dist, prop_dist, dataset_info


def get_dataloader(args_gen):
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
    return dataloaders


class DiffusionDataloader:
    def __init__(self, args_gen, model, nodes_dist, prop_dist, device, unkown_labels=False,
                 batch_size=1, iterations=200):
        self.args_gen = args_gen
        self.model = model
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.batch_size = batch_size
        self.iterations = iterations
        self.device = device
        self.unkown_labels = unkown_labels
        self.dataset_info = get_dataset_info(self.args_gen.dataset, self.args_gen.remove_h)
        self.i = 0

    def __iter__(self):
        return self

    def sample(self):
        nodesxsample = self.nodes_dist.sample(self.batch_size)
        context = self.prop_dist.sample_batch(nodesxsample).to(self.device)

        # Commenting to make the same as how EEGSDE handles it. No special handling for the other property        
        # property_to_control = args.property 
        # property_index = self.prop_dist.properties.index(property_to_control)
        # property_skip = [i for i in range(len(self.prop_dist.properties)) if i != property_index]
        # context[:, property_skip] = 0

        one_hot, charges, x, node_mask = sample(self.args_gen, self.device, self.model,
                                                self.dataset_info, self.prop_dist, nodesxsample=nodesxsample,
                                                context=context)

        node_mask = node_mask.squeeze(2)
        context = context.squeeze(1)

        # edge_mask
        bs, n_nodes = node_mask.size()
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        diag_mask = diag_mask.to(self.device)
        edge_mask *= diag_mask
        edge_mask_flattened = edge_mask.view(bs * n_nodes * n_nodes, 1)

        # Instead of just applying the transform to the first property, we apply it to all properties
        # prop_key = self.prop_dist.properties[0]
        if len(self.prop_dist.properties) > 1:
            for i in range(len(self.prop_dist.properties)):
                key = self.prop_dist.properties[i]
                context[:, i] = context[:, i] * self.prop_dist.normalizer[key]['mad'] + self.prop_dist.normalizer[key]['mean']
            # context_unorm[key] = context_i
        else:
            prop_key = self.prop_dist.properties[0]
            if self.unkown_labels:
                context[:] = self.prop_dist.normalizer[prop_key]['mean']
            else:
                context = context * self.prop_dist.normalizer[prop_key]['mad'] + self.prop_dist.normalizer[prop_key]['mean']
        # for prop in self.prop_dist.properties:
        #     if self.unkown_labels:
        #         context[:] = self.prop_dist.normalizer[prop]['mean']
        #     else:
        #         context = context * self.prop_dist.normalizer[prop]['mad'] + self.prop_dist.normalizer[prop]['mean']
        # breakpoint()
        data = {
            'positions': x.detach(),
            'atom_mask': node_mask.detach(),
            'edge_mask_unflattened': edge_mask.detach(),
            'edge_mask': edge_mask_flattened.detach(),
            'one_hot': one_hot.detach(),
            args.property : context.detach()
        }
        return data

    def __next__(self):
        if self.i <= self.iterations:
            self.i += 1
            return self.sample()
        else:
            self.i = 0
            raise StopIteration

    def __len__(self):
        return self.iterations


def main_quantitative(args):
    # Get classifier
    #if args.task == "numnodes":
    #    class_dir = args.classifiers_path[:-6] + "numnodes_%s" % args.property
    #else:
    class_dir = args.classifiers_path
    classifier = get_classifier(class_dir).to(args.device)

    # Get generator and dataloader used to train the generator and evalute the classifier
    args_gen = get_args_gen(args.generators_path)
    args_gen.fp_conditioning = False

    # Careful with this -->
    if not hasattr(args_gen, 'diffusion_noise_precision'):
        args_gen.normalization_factor = 1e-4
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'

    dataloaders = get_dataloader(args_gen)
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    if args.override_guidance:
        args_gen.guidance_weight = args.override_guidance

    
    model, nodes_dist, prop_dist, _ = get_generator(args.generators_path, dataloaders,
                                                    args.device, args_gen, property_norms, ckpt=args.ckpt)

    # Create a dataloader with the generator

    mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

    if args.task == 'edm':
        diffusion_dataloader = DiffusionDataloader(args_gen, model, nodes_dist, prop_dist,
                                                   args.device, batch_size=args.batch_size, iterations=args.iterations)
        print("EDM: We evaluate the classifier on our generated samples")
        loss = test(classifier, 0, diffusion_dataloader, mean, mad, args.property, args.device, 1, args.debug_break, 
                    args.use_wandb, args.exp_name, args.use_multiprop)
        print("Loss classifier on Generated samples: %.4f" % loss)
    elif args.task == 'qm9_second_half':
        print("qm9_second_half: We evaluate the classifier on QM9")
        loss = test(classifier, 0, dataloaders['train'], mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break, args.use_wandb, args.exp_name, args.use_multiprop)
        print("Loss classifier on qm9_second_half: %.4f" % loss)
    elif args.task == 'naive':
        print("Naive: We evaluate the classifier on QM9")
        length = dataloaders['train'].dataset.data[args.property].size(0)
        idxs = torch.randperm(length)
        dataloaders['train'].dataset.data[args.property] = dataloaders['train'].dataset.data[args.property][idxs]
        loss = test(classifier, 0, dataloaders['train'], mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break, args.use_wandb, args.exp_name, args.use_multiprop)
        print("Loss classifier on naive: %.4f" % loss)
    #elif args.task == 'numnodes':
    #    print("Numnodes: We evaluate the numnodes classifier on EDM samples")
    #    diffusion_dataloader = DiffusionDataloader(args_gen, model, nodes_dist, prop_dist, device,
    #                                               batch_size=args.batch_size, iterations=args.iterations)
    #    loss = test(classifier, 0, diffusion_dataloader, mean, mad, args.property, args.device, 1, args.debug_break)
    #    print("Loss numnodes classifier on EDM generated samples: %.4f" % loss)


def high_level_overview_old(exp_name, context, predicted_prop):
        # write a function that logs gen_properties to a csv file in the folder "outputs/%s/analysis/guidance/"
        base_folder = f"outputs/analysis/{exp_name}/guidance/run/"
        os.makedirs(base_folder, exist_ok=True)
        csv_file_path = os.path.join(base_folder, "gen_properties.csv")
        
        context_numpy = context.cpu().numpy()
        context = context_numpy[0]
        predicted_prop = predicted_prop.cpu().detach().numpy()
        gen_properties = {
            'True': context,
            'Predicted': predicted_prop,
        }

        file_exists = os.path.isfile(csv_file_path)

        # Open the CSV file for appending
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=gen_properties.keys())
            
            # Write the header only if the file is new
            if not file_exists:
                writer.writeheader()
            
            # Write the gen_properties data
            writer.writerow(gen_properties)

def micro_writes_old(exp_name, context, predicted_prop, run_id, ws):
    base_folder = f"outputs/analysis/{exp_name}/guidance/run{run_id}/"
    os.makedirs(base_folder, exist_ok=True)
    csv_file_path = os.path.join(base_folder, "gen_properties.csv")

    context_numpy = context.cpu().numpy()
    true_value = context_numpy[0][0]
    predicted_prop = predicted_prop.cpu().detach().numpy()

    # Ensure 'True' values array matches the length of 'Predicted' values
    true_values = np.full_like(predicted_prop, true_value)

    gen_properties = {
        'True': true_values,
        'Predicted': predicted_prop,
        'w': ws,
    }

    # Check if the CSV file exists to decide on writing headers
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file for appending
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file is new
        if not file_exists:
            writer.writerow(['True', 'Predicted', 'w'])
        
        # Write the 'True' and 'Predicted' values
        for i, (true_val, pred_val) in enumerate(zip(true_values, predicted_prop)):
            writer.writerow([true_val, pred_val, ws[i]])

def high_level_overview(exp_name, context, predicted_prop, run_id, ws):
    base_folder = f"outputs/analysis/{exp_name}/guidance/run/"
    os.makedirs(base_folder, exist_ok=True)
    csv_file_path = os.path.join(base_folder, "gen_properties.csv")
    
    context_numpy = context.cpu().numpy()
    true_value = context_numpy[0][0]
    predicted_prop = predicted_prop.cpu().detach().numpy()

    
    # Create a DataFrame
    df = pd.DataFrame({
        'ws': ws,
        'Run Id': [run_id] * len(predicted_prop),
        'True': [true_value] * len(predicted_prop),
        'Predicted': predicted_prop
    })

    df['mae'] = df.apply(lambda row: abs(row['True'] - row['Predicted']), axis=1)
    
    # Write to CSV, append if it exists
    df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)

def micro_writes(exp_name, context, predicted_prop, run_id, ws):
    base_folder = f"outputs/analysis/{exp_name}/guidance/run{run_id}/"
    os.makedirs(base_folder, exist_ok=True)
    csv_file_path = os.path.join(base_folder, "gen_properties.csv")
    
    context_numpy = context.cpu().numpy()
    true_value = context_numpy[0][0]
    predicted_prop = predicted_prop.cpu().detach().numpy()
    
    # Ensure 'True' values array matches the length of 'Predicted' values
    true_values = [true_value] * len(predicted_prop)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'True': true_values,
        'Predicted': predicted_prop,
        'w': ws
    })

    df['mae'] = df.apply(lambda row: abs(row['True'] - row['Predicted']), axis=1)
    
    # Write to CSV, append if it exists
    df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)

def longest_decreasing_subsequence_ids(maes):
    n = len(maes)
    # Initialize the LDS array and the predecessor array
    lds = [1] * n
    prev = [-1] * n

    # Compute the LDS values and track predecessors
    for i in range(1, n):
        for j in range(i):
            if maes[i] < maes[j] and lds[i] < lds[j] + 1:
                lds[i] = lds[j] + 1
                prev[i] = j

    # Find the index of the maximum value in LDS
    max_length = max(lds)
    max_index = lds.index(max_length)

    # Reconstruct the LDS by backtracking through the prev array
    lds_ids = []
    current_index = max_index
    while current_index != -1:
        lds_ids.append(current_index)
        current_index = prev[current_index]

    lds_ids.reverse()  # Reverse to get the correct order
    lds_seq = [maes[id] for id in lds_ids]  # Get the MAE values for the LDS IDs

    return lds_seq, lds_ids, max_length

def generate_figs(exp_name, epoch):
    base_folder = f"outputs/analysis/{exp_name}/guidance/run{epoch}"
    csv_path = f"{base_folder}/gen_properties.csv"
    df = pd.read_csv(csv_path)

    maes = df['mae'].to_list()
    _, ids_with_decreasing_mae, _ = longest_decreasing_subsequence_ids(maes)
    ws = df['w'].to_list()

    # Iterate through MAEs to find where a decrease occurs
    # ids_with_decreasing_mae = [0]
    # for i in range(1, len(maes)):
    #     if maes[i] < maes[i - 1]:  # Check if current MAE is less than the previous MAE
    #         ids_with_decreasing_mae.append(i) 

    pngs = [file for file in os.listdir(base_folder) if '.png' in file]
    png_idx = [int(file[-7:-4]) for file in pngs]
    pngs_sorted_by_indx = [x[1] for x in sorted(zip(png_idx, pngs))]

    filtered_pngs = [png for i, png in enumerate(pngs_sorted_by_indx) if i in ids_with_decreasing_mae]
    ws = [ws[i] for i in range(len(ws)) if i in ids_with_decreasing_mae]
    maes = [maes[i] for i in range(len(maes)) if i in ids_with_decreasing_mae]

    # Plot the filtered PNG files
    ncols = 5  # Adjust based on how many images per row you want
    nrows = (len(filtered_pngs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, nrows * (5+.5)))
    axes = axes.flatten()

    for idx, png in enumerate(filtered_pngs):
        img_path = os.path.join(base_folder, png)
        img = plt.imread(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')  # Hide axes
        title = f'$w$={ws[idx]}, MAE={maes[idx]:.2f}'
        axes[idx].set_title(title, fontsize=26) 

    # Hide unused subplots if any
    for ax in axes[len(filtered_pngs):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    save_path = os.path.join(base_folder, 'final/combined_images.png')
    os.makedirs(os.path.join(base_folder, 'final'), exist_ok=True)
    plt.savefig(save_path)

    plt.show()  # Display the plot after saving
    print(f"Figure saved to {save_path}")


def gen_prop_sweep_figs(exp_name, epoch, context):
    base_folder = f"outputs/prop_sweep/{exp_name}/run{epoch}"
    pngs = [file for file in os.listdir(base_folder) if '.png' in file]
    png_idx = [int(file[-7:-4]) for file in pngs]
    pngs_sorted_by_indx = [x[1] for x in sorted(zip(png_idx, pngs))]

    filtered_pngs = pngs_sorted_by_indx

    # Plot the filtered PNG files
    ncols = 5  # Adjust based on how many images per row you want
    nrows = (len(filtered_pngs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, nrows * (5+.5)))
    axes = axes.flatten()

    for idx, png in enumerate(filtered_pngs):
        img_path = os.path.join(base_folder, png)
        img = plt.imread(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')  # Hide axes
        title = f'{context[idx]:.2f}'
        axes[idx].set_title(title, fontsize=26) 

    # Hide unused subplots if any
    for ax in axes[len(filtered_pngs):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    save_path = f"outputs/prop_sweep/{exp_name}/final/{epoch}.png"
    os.makedirs(f"outputs/prop_sweep/{exp_name}/final", exist_ok=True)
    plt.savefig(save_path)

    plt.show()  # Display the plot after saving
    print(f"Figure saved to {save_path}")

def save_and_sample_guidance(classifier, args, device, model, prop_dist, dataset_info, epoch=0, id_from=0, seed=None):

    # breakpoint()
    # ws = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
    ws = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
    # ws = [0.5]
    one_hot_arr = []
    chargse_arr = []
    x_arr = []
    node_mask_arr = []
    context = []

    n_nodes = 19
    prop = args.conditioning[0]
    min_val, max_val = prop_dist.distributions[prop][n_nodes]['params']
    mean, mad = prop_dist.normalizer[prop]['mean'], prop_dist.normalizer[prop]['mad']
    min_val = (min_val - mean) / (mad)
    max_val = (max_val - mean) / (mad)
    context = torch.tensor((min_val+max_val)/2).float().to(device).repeat(1).unsqueeze(1)

    print(seed)
    for w in ws:
        model.w = w
        one_hot, charges, x, node_mask = sample_sweep_guidance(args, device, model, dataset_info, prop_dist, context, seed=seed)
        one_hot_arr.append(one_hot)
        chargse_arr.append(charges)
        x_arr.append(x)
        node_mask_arr.append(node_mask)

    node_mask = torch.cat(node_mask_arr, dim=0)
    node_mask = node_mask.squeeze(-1)
    node_mask_unsqueezed = node_mask
    bs, n_nodes = node_mask.size()

    one_hot = torch.cat(one_hot_arr, dim=0)
    charges = torch.cat(chargse_arr, dim=0)
    x = torch.cat(x_arr, dim=0)
    x_orig = x
    x = x.view(bs * n_nodes, -1).to(device, torch.float32)
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    node_mask = node_mask.view(bs * n_nodes, -1).to(device, torch.float32)
    edges = prop_utils.get_adj_matrix(n_nodes, bs, device)
    nodes = one_hot.to(device, torch.float32)
    nodes = nodes.view(bs * n_nodes, -1).to(device, torch.float32)
    edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1).to(device, torch.float32)
    predicted_prop = classifier(h0=nodes, x=x, edges=edges, edge_attr=None, node_mask=node_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)
    
    
    vis.save_xyz_file(
        'outputs/analysis/%s/guidance/run%s/' % (args.exp_name, epoch), one_hot, charges, x_orig, dataset_info,
        id_from, name='conditional', node_mask=node_mask_unsqueezed)

    vis.visualize_chain("outputs/analysis/%s/guidance/run%s/" % (args.exp_name, epoch), dataset_info,
                        wandb=None, mode='conditional', spheres_3d=True)
    

    high_level_overview(args.exp_name, context, predicted_prop, epoch, ws)

    micro_writes(args.exp_name, context,predicted_prop, epoch, ws)

    generate_figs(args.exp_name, epoch)

    return one_hot, charges, x_orig

def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0, seed=None):
    n_nodes = 19
    n_frames = 10
    context = []
    for key in prop_dist.distributions:
        min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
        mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
        min_val = (min_val - mean) / (mad)
        max_val = (max_val - mean) / (mad)
        context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
        print(np.linspace(min_val, max_val, n_frames))
        context.append(context_row)
    context = torch.cat(context, dim=1).float().to(device)
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist, context=context, seed=seed, n_frames=n_frames)

    base_folder = 'outputs/prop_sweep/%s/run%s/' % (args.exp_name, epoch)
    vis.save_xyz_file(
        base_folder, one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    vis.visualize_chain(base_folder, dataset_info,
                        wandb=None, mode='conditional', spheres_3d=True)

    csv_folder = 'outputs/prop_sweep/%s/' % (args.exp_name)
    # sample every 10 entries from context
    context = context.cpu().detach().numpy().flatten()
    # save context to csv
    context_df = pd.DataFrame(context)
    os.makedirs(csv_folder, exist_ok=True)
    context_df.to_csv(csv_folder + 'context.csv', index=False, header=False)

    # context_sampled = context[::10].cpu().detach().numpy()
    # one_hot_sampled = one_hot[::10].cpu().detach().numpy()
    # charges_sampled = charges[::10].cpu().detach().numpy()
    # x_sampled = x[::10].cpu().detach().numpy()
    # node_mask_sampled = node_mask[::10].cpu().detach().numpy()
   
    
    
    
    gen_prop_sweep_figs(args.exp_name, epoch, context)



    return one_hot, charges, x


def main_qualitative(args):
    
    class_dir = args.classifiers_path
    classifier = get_classifier(class_dir).to(args.device)

    args_gen = get_args_gen(args.generators_path)
    if args.override_guidance:
        args_gen.guidance_weight = args.override_guidance
    args_gen.ckpt = args.ckpt
    args_gen.fp_conditioning = False
    dataloaders = get_dataloader(args_gen)
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    model, nodes_dist, prop_dist, dataset_info = get_generator(args.generators_path,
                                                               dataloaders, args.device, args_gen,
                                                               property_norms, ckpt=args_gen.ckpt)

    
    if args.mode == 'guidance':
        for i in range(args.n_sweeps):
            print("Sampling sweep for %d/%d" % (i+1, args.n_sweeps))
            save_and_sample_guidance(classifier, args_gen, device, model, prop_dist, dataset_info, epoch=i, id_from=0, seed=i)

    else:
        for i in range(args.n_sweeps):
            print("Sampling sweep %d/%d" % (i+1, args.n_sweeps))
            save_and_sample_conditional(args_gen, device, model, prop_dist, dataset_info, epoch=i, id_from=0, seed=i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug_alpha')
    parser.add_argument('--generators_path', type=str, default='outputs/exp_cond_alpha_pretrained')
    parser.add_argument('--classifiers_path', type=str, default='qm9/property_prediction/outputs/exp_class_alpha_pretrained')
    parser.add_argument('--property', type=str, default='alpha',
                        help="'alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv'")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--debug_break', type=eval, default=False,
                        help='break point or not')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='break point or not')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='break point or not')
    parser.add_argument('--iterations', type=int, default=20,
                        help='break point or not')
    parser.add_argument('--task', type=str, default='qualitative',
                        help='naive, edm, qm9_second_half, qualitative')
    parser.add_argument('--n_sweeps', type=int, default=10,
                        help='number of sweeps for the qualitative conditional experiment')
    parser.add_argument('--override_guidance', type=float, help='Whether or not to override the guidance weight parameter')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging of classifier')
    parser.add_argument('--use_multiprop', action='store_true', help="Classifier is being run on a multiproperty conditional model")
    parser.add_argument('--ckpt', type=int, default=None, help='Checkpoint number to load')
    parser.add_argument('--mode', type=str, default='property')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    
    if args.task == 'qualitative':
        main_qualitative(args)
    else:
        main_quantitative(args)
