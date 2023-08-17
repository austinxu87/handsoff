import os
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    args = parser.parse_args()

    args = json.load(open(args.exp, 'r'))

    assert 'uncertainty_portion' in args, f"uncertainty_portion needs to be included in JSON file. Some JSON files were not updated accordingly (sorry!)"

    base_path = "deeplab_class_%d_checkpoint_%d_filter_out_%f_merge_classes_%s_num_train_%0d" %(args['testing_data_number_class'],
                                                                                        args['num_synth_train'],
                                                                                        args['uncertainty_portion'],
                                                                                        str(args['merge_classes']),
                                                                                        args['num_synth_train'])
    print(base_path)
    print(args['uncertainty_portion'])

