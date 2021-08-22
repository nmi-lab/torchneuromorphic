#!/usr/bin/env python
# Copyright Friedemann Zenke 2020

import argparse
import json
import numpy as np
import sys
import os
import time
import logging

# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

import torch
import torchvision

# sys.path.append(os.path.expanduser("~/projects/randman"))
# import randman

import stork
import stork.datasets
from stork.models import RecurrentSpikingModel
from stork.generators import StandardGenerator
from stork.nodes import InputGroup, ReadoutGroup, LIFGroup
from stork.connections import Connection

import shared.network


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("netimg")

    parser = argparse.ArgumentParser("Perform single classification experiment")
    parser.add_argument("--nb_inputs", type=int, default=28*28,
                        help="Number of input nodes")
    parser.add_argument("--nb_classes", type=int, default=10,
                        help="Number of output nodes")
    parser.add_argument("--nb_hidden_units", type=int, default=100,
                        help="Number of nodes in hidden layer")
    parser.add_argument("--nb_hidden_layers", type=int, default=1,
                        help="Number of layers in network")
    parser.add_argument("--dataset", type=str, default="mnist",
                        help="Name of image dataset")
    parser.add_argument("--validation_split", type=float,
                        help="Validation split ratio", default=1.0)
    parser.add_argument("--generator_time_scale", type=float, default=50e-3,
                        help="Characteristic timescale of generator in ms")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for network initialization")
    parser.add_argument("--recurrent", action='store_true', default=False,
                        help="Allow recurrent synapses")
    parser.add_argument("--detach", action='store_true', default=False,
                        help="Stop gradients from flowing through recurrent connections")
    parser.add_argument("--rec_winit", type=float, default=1.0,
                        help="Recurrent weight init scale of stdev.")
    parser.add_argument("--tau_mem", type=float, default=10e-3,
                        help="Membrane time constant of LIF neurons")
    parser.add_argument("--tau_readout", type=float, default=20e-3,
                        help="Membrane time constant of readout neurons")
    parser.add_argument("--beta", type=float, default=100.0,
                        help="Steepness parameter of the surrogate gradient")
    parser.add_argument("--tau_syn", type=float, default=5e-3,
                        help="Synaptic time constant of LIF neurons")
    parser.add_argument("--time_step", type=float, default=1e-3,
                        help="Integration time step in ms")
    parser.add_argument("--nb_time_steps", type=int, default=100,
                        help="Number of simulation time steps")
    parser.add_argument("--batch_size", type=int, default=250,
                        help="Number of digits in a single batch")
    parser.add_argument("--nb_epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate")
    parser.add_argument("--activation", type=str, default="SuperSpike",
                        help="Activation function")
    parser.add_argument("--learntau", action='store_true',
                        help="Learn neuronal timescales")
    parser.add_argument("--upperBoundL1Strength", type=float, default=0.06,
                        help="Upper bound L1 regulizer strength")
    parser.add_argument("--upperBoundL1Threshold", type=float, default=100.0,
                        help="Upper bound L1 regulizer threshold")
    parser.add_argument("--upperBoundL2Strength", type=float, default=0.0,
                        help="Upper bound L2 regulizer strength")
    parser.add_argument("--upperBoundL2Threshold", type=float, default=100.0,
                        help="Upper bound L2 regulizer threshold")
    parser.add_argument("--lowerBoundL2Threshold", type=float, default=1e-3,
                        help="Lower bound L2 regulizer threshold")
    parser.add_argument("--lowerBoundL2Strength", type=float, default=100.0,
                        help="Lower bound L2 regulizer threshold")
    parser.add_argument("--weightL2Strength", type=float, default=0.0,
                        help="Weight L2 regulizer strength")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Probability of dropping a hidden layer spike")
    parser.add_argument("--p_drop", type=float, default=0.0,
                        help="Probability of dropping an input spike")
    parser.add_argument("--p_insert", type=float, default=0.0,
                        help="Probability of inserting an input spike")
    parser.add_argument("--sigma_t", type=float, default=0.0,
                        help="Time jitter amplitude added to spikes in bins")
    parser.add_argument("--sigma_u", type=int, default=0.0,
                        help="Unit jitter amplitude added to spikes")
    parser.add_argument("--gpu", action='store_true',
                        help="Run simulation on gpu")
    parser.add_argument("--record_spikes", action='store_true',
                        help="Record spikes")
    parser.add_argument("--bias", action='store_true',
                        help="Use bias term in forward connections except readout")
    parser.add_argument("--nb_threads", type=int, default=None,
                        help="Number of threads")
    parser.add_argument("--nb_workers", type=int, default=2,
                        help="Number of independent dataloader worker threads")
    parser.add_argument("--diffreset", action='store_true', default=False,
                        help="Enable differentiating reset term")
    parser.add_argument("--loss_type", type=str, default="MaxOverTime",
                        help="Select loss type")
    parser.add_argument("--save", action='store_true', default=False, 
                        help="If set the model state is saved stored.")
    parser.add_argument("--plot", action='store_true', default=False, 
                        help="If set, some plots are saved.")
    parser.add_argument("--load", type=str, default="",
                        help="Filepath to statefile to load network state")
    parser.add_argument("--prefix", type=str, default="img",
                        help="Output file prefix")
    parser.add_argument("--timestr", type=str, default=None,
                        help="Timestr for filename")
    parser.add_argument("--gitdescribe", type=str, default="default",
                        help="Git describe string for logging")
    parser.add_argument("--dir", type=str, default="out",
                        help="Path to where to write results as json")
    parser.add_argument("--notest", action='store_true', default=False)
    parser.add_argument("--verbose", action='store_true', default=False)

    args = parser.parse_args()



    sim_start_time = time.time()
    # store datetime string
    if args.timestr is None: 
        timestr = time.strftime("%Y%m%d-%H%M%S")
        basepath = stork.utils.get_basepath(args.dir,args.prefix)
    else:
        timestr = args.timestr
        basepath = "%s/%s-%s"%(args.dir,args.prefix,timestr)

    # store datetime string
    results = dict(datetime=timestr, basepath=basepath, args=vars(args))


    # create error file handler and set level to error
    log_filename = "%s.log"%basepath
    file_handler = logging.FileHandler(log_filename, "w", encoding=None, delay="true")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
        logger.debug("Set loglevel: DEBUG")
    else:
        logger.setLevel(logging.INFO)

    logger.info("Basepath {}".format(basepath))


    if args.plot:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt


    dtype = torch.float
    if args.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available(): 
            logger.warning("Cuda is available, but not used.")



    if args.nb_threads is not None:
        logger.debug("PyTorch set to use %i threads."%args.nb_threads)
        torch.set_num_threads(args.nb_threads)


    logger.info("Loading %s dataset"%args.dataset)
    if args.dataset == "mnist":
        root = os.path.expanduser("/tungstenfs/scratch/gzenke/zenkfrie/data/datasets/torch/mnist")
        train_dataset = torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=True)
        test_dataset  = torchvision.datasets.MNIST(root, train=False, transform=None, target_transform=None, download=True)
    elif args.dataset == "fashion-mnist":
        root = os.path.expanduser("/tungstenfs/scratch/gzenke/zenkfrie/data/datasets/torch/fashion-mnist")
        train_dataset = torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=True)
        test_dataset  = torchvision.datasets.FashionMNIST(root, train=False, transform=None, target_transform=None, download=True)
    else:
        raise NotImplemented 
    # TODO implement


    ds_kwargs = dict(
                    nb_steps=args.nb_time_steps,
                    nb_units=args.nb_inputs,
                    tau=args.generator_time_scale,
                    p_drop=args.p_drop,
                    p_insert=args.p_insert,
                    sigma_t=args.sigma_t,
                    # sigma_u=args.sigma_u,
                    )

    if args.validation_split == 1.0:
        tmp_train = train_dataset.data.reshape((-1,28*28)).float()/255, train_dataset.targets

        ds_train = stork.datasets.SpikeLatencyDataset(tmp_train, **ds_kwargs)
    else:
        tmp_train, tmp_valid = stork.datasets.split_dataset(train_dataset.data.reshape((-1,28*28)).float()/255, train_dataset.targets,
                                                           splits=[args.validation_split, 1.0-args.validation_split ], shuffle=False)

        ds_train = stork.datasets.SpikeLatencyDataset(tmp_train, **ds_kwargs)
        ds_valid = stork.datasets.SpikeLatencyDataset(tmp_valid, **ds_kwargs)

    tmp_test = test_dataset.data.reshape((-1,28*28)).float()/255, test_dataset.targets
    ds_test = stork.datasets.SpikeLatencyDataset(tmp_test, **ds_kwargs)


    logger.debug("Loaded dataset with {} classes".format(args.nb_classes))


    if args.seed is not None:
        torch.random.manual_seed(args.seed)

    logger.info("Setting up model ...")
    model = shared.network.get_model(args, device, dtype)



    # Monitors spike counts before training
    res = model.monitor(ds_test)
    total_spikes_per_layer = [  torch.sum(res[i]).item() for i in range(args.nb_hidden_layers) ]
    results["total_layer_spikes_init"] = total_spikes_per_layer
    results["avg_layer_spikes_per_input_init"] = [ nb/len(ds_test) for nb in total_spikes_per_layer ]


    if args.nb_epochs:
        logger.info("Training model on training data...")

        if args.validation_split == 1.0:
            history = model.fit( ds_train, 
                               nb_epochs=args.nb_epochs, 
                               verbose=args.verbose)
        else:
            history = model.fit_validate( ds_train, ds_valid,
                               nb_epochs=args.nb_epochs, 
                               verbose=args.verbose)
            results["valid_loss"] = history["loss_val"].tolist()
            results["valid_acc"]  = history["acc_val"].tolist()
            logger.info("Valid loss %.3f"%history["loss_val"][-1])
            logger.info("Valid acc  %.3f"%history["acc_val"][-1])

        results["train_loss"] = history["loss"].tolist()
        results["train_acc"]  = history["acc"].tolist()
        logger.info("Train loss %.3f"%history["loss"][-1])
        logger.info("Train acc  %.3f"%history["acc"][-1])



    # Monitors spike counts after training
    res = model.monitor(ds_test)
    total_spikes_per_layer = [  torch.sum(res[i]).item() for i in range(args.nb_hidden_layers) ]
    results["total_layer_spikes"] = total_spikes_per_layer
    results["avg_layer_spikes_per_input"] = [ nb/len(ds_test) for nb in total_spikes_per_layer ]

    if args.record_spikes:
        for i in range(args.nb_hidden_layers):
            filepath = "%s-l%i.npy"%(basepath,i)
            coords = np.argwhere(res[args.nb_hidden_layers+i].cpu().numpy())
            np.save(filepath,coords)

    if not args.notest:
        logger.info("Evaluating model on test data ...")
        scores = model.evaluate(ds_test).tolist()
        results["test_loss"], _,results["test_acc"] = scores
        logger.info("Test loss %.3f"%scores[0])
        logger.info("Test acc  %.3f"%scores[2])

   
    # Store wall clock time duration
    sim_end_time = time.time()
    results["wall_time"] = sim_end_time - sim_start_time

    logger.debug("Saving results")
    filepath = "%s.json"%(basepath)
    with open(filepath, "w") as file_handle:
        json.dump(results, file_handle, indent=4)


    if args.plot:
        filepath = "%s-out.png"%(basepath)
        stork.plotting.plot_activity_snapshot(model,
                                              nb_samples=4,
                                              point_alpha=0.3)
        plt.savefig(filepath,dpi=150)

    if args.save:
        logger.debug("Saving model")
        filepath = "%s.state"%(basepath)
        torch.save(model.state_dict(), filepath)
