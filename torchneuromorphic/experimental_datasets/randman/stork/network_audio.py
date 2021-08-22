#!/usr/bin/env python
# Copyright Friedemann Zenke 2020

import argparse
import json
import numpy as np
import sys
import os
import time
import logging

import torch
import stork
import stork.datasets
from stork.models import RecurrentSpikingModel
from stork.datasets import RawHeidelbergDigits, RawSpeechCommands, DatasetView
from stork.generators import StandardGenerator
from stork.nodes import InputGroup, InputWarpGroup, ReadoutGroup, ControlLIGroup, LIFGroup, DualSynLIFGroup
from stork.connections import Connection


import shared.network



def speaker_performance(y, y_pred, ids):
    num_ids = np.unique(ids).size
    speaker_acc = np.empty(num_ids)
    for i in range(num_ids):
        idx = (ids==i)
        speaker_acc[i] = (y[idx]==y_pred[idx]).mean()
    return speaker_acc


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("netraw")
    
    parser = argparse.ArgumentParser("Perform single classification experiment")
    parser.add_argument("--nb_inputs", type=int, default=40,
                        help="Number of input nodes")
    parser.add_argument("--dataset", type=str,
                        help="Name of dataset", default="hd")
    parser.add_argument("--training_data", type=str,
                        help="Path to training data", default=None)
    parser.add_argument("--validation_split", type=float,
                        help="Validation split ratio", default=0.0)
    parser.add_argument("--nb_input_steps", type=int, default=80,
                        help="Number of input time steps")
    parser.add_argument("--testing_data", type=str,
                        help="Path to testing data", default=None)
    parser.add_argument("--diffcode", action='store_true', default=False,
                        help="Use event-based code for input")
    parser.add_argument("--keywordspotting", action='store_true', default=False,
                        help="Perform keyword spotting")
    parser.add_argument("--nb_classes", type=int, default=35,
                        help="Number of output classes")
    parser.add_argument("--nb_hidden_units", type=int, default=128,
                        help="Number of nodes in hidden layer")
    parser.add_argument("--nb_hidden_layers", type=int, default=1,
                        help="Number of layers in network")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for network initialization")
    parser.add_argument("--recurrent", action='store_true', default=False,
                        help="Allow recurrent synapses")
    parser.add_argument("--detach", action='store_true', default=False,
                        help="Stop gradients from flowing through recurrent connections")
    parser.add_argument("--rec_winit", type=float, default=1.0,
                        help="Recurrent weight init scale of stdev.")
    parser.add_argument("--tau_mem", type=float, default=20e-3,
                        help="Membrane time constant of LIF neurons")
    parser.add_argument("--stp", action='store_true', default=False,
                        help="Use short-term plasticity on recurrent connections")
    parser.add_argument("--tau_readout", type=float, default=20e-3,
                        help="Membrane time constant of readout neurons")
    parser.add_argument("--beta", type=float, default=100.0,
                        help="Steepness parameter of the surrogate gradient")
    parser.add_argument("--tau_syn", type=float, default=10e-3,
                        help="Synaptic time constant of LIF neurons")
    parser.add_argument("--time_step", type=float, default=2e-3,
                        help="Integration time step in ms")
    parser.add_argument("--duration", type=float, default=0.8,
                        help="Maximum duration to consider")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Number of digits in a single batch")
    parser.add_argument("--nb_epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
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
                        help="Lower bound L2 regulizer strength")
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
    parser.add_argument("--bias", action='store_true',
                        help="Use bias term in forward connections except readout")
    parser.add_argument("--nb_threads", type=int, default=None,
                        help="Number of threads")
    parser.add_argument("--nb_workers", type=int, default=3,
                        help="Number of independent dataloader worker threads")
    parser.add_argument("--diffreset", action='store_true', default=False,
                        help="Enable differentiating reset term")
    parser.add_argument("--loss_type", type=str, default="MaxOverTime",
                        help="Select loss type")
    parser.add_argument("--prefix", type=str, default="raw",
                        help="Output file prefix")
    parser.add_argument("--timestr", type=str, default=None,
                        help="Timestr for filename")
    parser.add_argument("--gitdescribe", type=str, default="default",
                        help="Git describe string for logging")
    parser.add_argument("--dir", type=str, default="out",
                        help="Path to where to write results as json")
    parser.add_argument("--cachedir", type=str, default="/tungstenfs/scratch/gzenke/datasets/cache/",
                        help="Path to cache directory")
    parser.add_argument("--save", action='store_true',
                        help="Save results as json")
    parser.add_argument("--plot", action='store_true', default=False, 
                        help="If set, some plots are saved.")
    parser.add_argument("--load", type=str, default=None,
                        help="Filepath to statefile to load network state")
    parser.add_argument("--speaker_acc", action='store_true',
                        help="Compute per-speaker accuracy on test set")
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

    if args.seed is not None:
        torch.random.manual_seed(args.seed)


    logger.info("Building model")
    args.nb_time_steps = int(args.duration/args.time_step)
    logger.debug("Simulation with %i time steps"%args.nb_time_steps)
    nb_input_steps = args.nb_input_steps 
    logger.debug("Simulation with %i input time steps"%nb_input_steps)

    model = shared.network.get_model(args, device, dtype)



    if args.load:
        logger.debug("Loading network state from file")
        model.load_state_dict(torch.load(args.load))
    
    gen_kwargs = dict( nb_steps=nb_input_steps,
                       time_step=10e-3,
                       diffcode=args.diffcode, 
                      # time_scale=1.0/args.time_step,
                      # nb_units=args.nb_inputs,
                      # p_insert=args.p_insert,
                      # p_drop=args.p_drop,
                      # sigma_t=args.sigma_t,
                      # sigma_u=args.sigma_u
                      )

    if args.keywordspotting:
        gen_kwargs["label_mode"] = "keyword spotting"


    if args.training_data:
        logger.info("Load data")
        if args.dataset=="hd":

            def read_filelist(filename):
                with open(filename) as f:
                    content = f.readlines()
                # Remove whitespace characters like `\n` at the end of each line
                content = [x.strip() for x in content]
                return content

            dirpath=args.training_data
            train_dataset = RawHeidelbergDigits(dirpath, subset=read_filelist("%s/train_filenames.txt"%dirpath), cache_fname="%s/cache-%s-training.pkl.gz"%(args.cachedir, args.prefix), **gen_kwargs)
            logger.debug("Opened HD dataset with %i data"%len(train_dataset))
            test_dataset = RawHeidelbergDigits(dirpath, subset=read_filelist("%s/test_filenames.txt"%dirpath), cache_fname="%s/cache-%s-test.pkl.gz"%(args.cachedir, args.prefix), **gen_kwargs)
            logger.debug("Opened HD dataset with %i data"%len(test_dataset))

            # mother_dataset=train_dataset
            # elements = np.arange(len(mother_dataset))
            # np.random.shuffle(elements)
            # split = int(0.9*len(mother_dataset))
            # logger.debug("Split at %i"%(split))
            # train_dataset = DatasetView(mother_dataset, elements[:split])
            # test_dataset = DatasetView(mother_dataset, elements[split:])
            # logger.debug("Split off %i test data"%len(test_dataset))
            
            if args.validation_split:
                logger.info("Splitting off validation data")
                mother_dataset=train_dataset
                elements = np.arange(len(mother_dataset))
                np.random.shuffle(elements)
                split = int(args.validation_split*len(mother_dataset))
                train_dataset = DatasetView(mother_dataset, elements[:split])
                valid_dataset = DatasetView(mother_dataset, elements[split:])

        elif args.dataset=="sc":
            args.validation_split = 0.9 # Since this is fixed for this dataset
            logger.debug("Loading training data")
            train_dataset = RawSpeechCommands(args.training_data, subset="training", cache_fname="%s/SCcache-%s-training.pkl.gz"%(args.cachedir, args.prefix), **gen_kwargs)
            logger.debug("Loading validation data")
            valid_dataset = RawSpeechCommands(args.training_data, subset="validation", cache_fname="%s/SCcache-%s-validation.pkl.gz"%(args.cachedir, args.prefix), **gen_kwargs)
            logger.debug("Loading test data")
            test_dataset = RawSpeechCommands(args.training_data, subset="testing", cache_fname="%s/SCcache-%s-testing.pkl.gz"%(args.cachedir, args.prefix), **gen_kwargs)

        logger.info("Loaded %i training data"%len(train_dataset))
        if args.validation_split: logger.info("Loaded %i validation data"%len(valid_dataset))
        logger.info("Loaded %i test data"%len(test_dataset))

            # valid_dataset.p_drop = 0.0
            # valid_dataset.p_insert = 0.0
       
        if args.nb_epochs: 
            logger.info("Fitting model to train data")
            if args.validation_split:
                history = model.fit_validate(train_dataset, valid_dataset,
                                             nb_epochs=args.nb_epochs,
                                             verbose=args.verbose)
            else:
                history = model.fit(train_dataset,
                                             nb_epochs=args.nb_epochs,
                                             verbose=args.verbose)

            results["train_loss"] = history["loss"].tolist()
            results["train_acc"]  = history["acc"].tolist()
            logger.info("Train loss {:.3f}".format(history["loss"][-1]))
            logger.info("Train acc {:.3f}".format(history["acc"][-1]))

            if args.validation_split:
                results["valid_loss"] = history["loss_val"].tolist()
                results["valid_acc"]  = history["acc_val"].tolist()
                logger.info("Valid loss {:.3f}".format(history["loss_val"][-1]))
                logger.info("Valid acc {:.3f}".format(history["acc_val"][-1]))


    logger.info("Evaluation model performance on test data")
    scores = model.evaluate(test_dataset).tolist()
    results["test_loss"] = scores[0]
    results["test_acc"] = scores[2]
    logger.info("Test loss {:.3f}".format(scores[0]))
    logger.info("Test acc {:.3f}".format(scores[2]))

    # Monitors spike counts before training
    layer_counts = model.monitor(test_dataset)
    total_spikes_per_layer = [  torch.sum(counts).item() for counts in layer_counts ]
    results["total_layer_spikes"] = total_spikes_per_layer
    results["avg_layer_spikes_per_input"] = [ nb/len(test_dataset) for nb in total_spikes_per_layer ]

   
    if args.speaker_acc:
        logger.info("Estimating per-speaker accuracy...")
        speaker_ids = test_dataset.h5file.root.extra.speaker
        y_pred = model.get_predictions(test_dataset)
        results["speaker_performance"] = speaker_performance(test_dataset.labels.cpu().numpy(),
                                                   y_pred, np.array(speaker_ids)).tolist()

    # basepath = "%s/%s-%s"%(args.dir, timestr, args.prefix)
    
    
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
