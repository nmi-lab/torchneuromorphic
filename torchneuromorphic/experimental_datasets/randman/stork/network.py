import torch
import stork
import stork.datasets
from stork.models import RecurrentSpikingModel
from stork.datasets import HDF5Dataset, DatasetView
from stork.generators import StandardGenerator
from stork.nodes import InputGroup, InputWarpGroup, ReadoutGroup, ControlLIGroup, LIFGroup, SNUGroup, DualSynLIFGroup, TsodyksMarkramSTP
from stork.connections import Connection
import stork.regularizers


def get_model(args, device, dtype, logger=None):

    if args.activation=="SuperSpike":
        act_fn = stork.activations.SuperSpike
    elif args.activation=="SigmoidSpike":
        act_fn = stork.activations.SigmoidSpike
    elif args.activation=="EsserSpike":
        act_fn = stork.activations.EsserSpike
    elif args.activation=="SuperSpike_asymptote":
        act_fn = stork.activations.SuperSpike_asymptote
    else:
        logger.error("Invalid activation function %s"%(args.activation))

    if not hasattr(args, 'stp'):
        args.stp = False

    if not hasattr(args, 'bias'):
        args.bias = False

    if not hasattr(args, 'rec_winit'):
        args.rec_winit = 1.0

    # Change beta for all SuperSpike nls
    act_fn.beta = args.beta

    neuron_group = LIFGroup 
    # neuron_group = SNUGroup # FIXME

    model = RecurrentSpikingModel(args.batch_size,
                                  args.nb_time_steps,
                                  args.nb_inputs,
                                  device,
                                  dtype)

    if hasattr(args, 'nb_input_steps'):
        input_group = model.add_group(InputWarpGroup(args.nb_inputs, args.nb_input_steps, scale=args.nb_input_steps/args.nb_time_steps))
    else:
        input_group = model.add_group(InputGroup(args.nb_inputs))

    if not hasattr(args, 'dropout'):
        args.dropout = None

    # Define regularizer list
    regs = []
    w_regs = []
    if args.lowerBoundL2Strength:
        reg1 = stork.regularizers.PerNeuronLowerBoundL2Regularizer(args.lowerBoundL2Strength,
                                                threshold=args.lowerBoundL2Threshold)
        regs.append(reg1)

    if hasattr(args, 'upperBoundL1Strength') and args.upperBoundL1Strength:
        reg = stork.regularizers.PopulationUpperBoundL1Regularizer(args.upperBoundL1Strength,
                                                 threshold=args.upperBoundL1Threshold)
        regs.append(reg)

    if hasattr(args, 'weightL2Strength') and args.weightL2Strength:
        w_reg = stork.regularizers.WeightL2Regularizer(args.weightL2Strength)
        w_regs.append(w_reg)

    if args.upperBoundL2Strength:
        reg2 = stork.regularizers.PopulationUpperBoundL2Regularizer(args.upperBoundL2Strength,
                                                 threshold=args.upperBoundL2Threshold)
        regs.append(reg2)


    # add cell groups
    upstream_group = input_group
    for l in range(args.nb_hidden_layers):
        neurons = model.add_group(LIFGroup(args.nb_hidden_units,
                                           tau_mem=args.tau_mem,
                                           tau_syn=args.tau_syn,
                                           diff_reset=args.diffreset,
                                           dropout_p=args.dropout,
                                           activation=act_fn))

        neurons.regularizers.extend(regs)

        con = model.add_connection(Connection(upstream_group, neurons, bias=args.bias))
        con.regularizers.extend(w_regs)

        
        neurons_in = neurons
        if args.recurrent:
            if args.stp:
                neurons = model.add_group(TsodyksMarkramSTP(neurons, tau_d=500e-3))
            con = model.add_connection(Connection(neurons, neurons_in, propagate_gradients=(not args.detach)))
            con.init_parameters(0.0,args.rec_winit) # scale down the recurrent strength  default gain is sqrt(5)
            con.regularizers.extend(w_regs)

        upstream_group=neurons

    readout_group = model.add_group(ReadoutGroup(args.nb_classes,
                                                 tau_mem=args.tau_readout,
                                                 tau_syn=args.tau_syn))
    con_exc = model.add_connection(Connection(upstream_group, readout_group))
   


    # Add spike count monitors
    for i in range(args.nb_hidden_layers):
        model.add_monitor(stork.monitors.SpikeCountMonitor(model.groups[1+i]))

    for i in range(args.nb_hidden_layers):
        model.add_monitor(stork.monitors.StateMonitor(model.groups[1+i],"out"))

    generator = StandardGenerator(nb_workers=args.nb_workers)

    # if args.last_step_loss:
    #     loss_stack = stork.loss_stacks.LastStepCrossEntropyReadoutStack()
    # else:
    #     loss_stack = stork.loss_stacks.TemporalCrossEntropyReadoutStack()

    if args.loss_type=="MaxOverTime":
        loss_stack = stork.loss_stacks.MaxOverTimeCrossEntropy()
    elif args.loss_type=="MaxOverTimeBinary":
        loss_stack = stork.loss_stacks.MaxOverTimeBinaryCrossEntropy()
    elif args.loss_type=="LastStep":
        loss_stack = stork.loss_stacks.LastStepCrossEntropy()
    elif args.loss_type=="SumOverTime":
        loss_stack = stork.loss_stacks.SumOverTimeCrossEntropy()
    else:
        logger.warning("Unknown loss type, defaulting to MaxOverTimeCrossEntropy")
        loss_stack = stork.loss_stacks.MaxOverTimeCrossEntropy()

    # Select optimizer
    # opt=torch.optim.Adam
    opt=stork.optimizers.SMORMS3
    # opt=torch.optim.Adamax

    model.configure(input=input_group,
                    output=readout_group,
                    loss_stack=loss_stack,
                    generator=generator,
                    optimizer=opt,
                    optimizer_kwargs=dict(lr=args.lr),
                    time_step=args.time_step)

    # Load network parameters from statefile if given
    if args.load:
        logger.debug("Loading network state from file")
        model.load_state_dict(torch.load(args.load))

    return model
