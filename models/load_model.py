import os

from models.hourglass import create_hourglass_network

def load_model(args, scales):
    print("Initializing model")
    print("Batch size: ", args.batch_size)
    print("Num stacks: ", args.num_stacks)
    print("Input size: {} x {}".format(args.input_size, args.input_size))
    print("Heatmap size: {} x {}".format(args.heatmap_size, args.heatmap_size))
    print("Training for {} epochs".format(args.epochs))
    print("Scales: ", scales)
    print("Channels: ", args.channels)
    print("Experiment number: ", args.experiment)
    print("Mobilenet version: ", args.mobilenet)

    if args.mobilenet:
        module = 'mobilenet'
        module_str = 'm'
    else:
        module = 'bottleneck'
        module_str = 'b'

    snapshot_dir_name = 'VP1VP2{}_{}in_{}out_{}s_{}n_{}b_{}c_{}'.format(module_str, args.input_size, args.heatmap_size,
                                                                        len(scales), args.num_stacks, args.batch_size,
                                                                        args.channels, args.experiment)
    snapshot_dir_path = os.path.join('snapshots', snapshot_dir_name)

    if not os.path.exists(snapshot_dir_path):
        os.makedirs(snapshot_dir_path)

    print("Checkpoint dir name: ", snapshot_dir_name)

    model = create_hourglass_network(2 * len(scales), args.num_stacks, inres=args.input_size, outres=args.heatmap_size,
                                     bottleneck=module, num_channels=args.channels)

    if args.resume:
        resume_model_path = os.path.join(snapshot_dir_path, 'model.{:03d}.h5'.format(args.resume))
        print("Loading model", resume_model_path)
        model.load_weights(resume_model_path)

    return model, snapshot_dir_name, snapshot_dir_path