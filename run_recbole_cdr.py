
import argparse
import os

from recbole_cdr.quick_start import run_recbole_cdr

if __name__ == '__main__':

    model_Name = 'DGCDR'
    # model_Name = 'BiTGCF'
    # model_Name = 'DCCDR'
    # model_Name = 'DTCDR'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default=model_Name, help='name of models')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()
    overall_config = 'recbole_cdr/properties/overall.yaml'

    # data_config = 'recbole_cdr/properties/dataset/AmazonElec_AmazonCloth_commonUser_10-core.yaml'
    # data_config = 'recbole_cdr/properties/dataset/AmazonCloth_AmazonElec_commonUser_10-core.yaml'
    # data_config = 'recbole_cdr/properties/dataset/AmazonSport_AmazonCloth_commonUser_5-core.yaml'
    # data_config = 'recbole_cdr/properties/dataset/AmazonCloth_AmazonSport_commonUser_5-core.yaml'
    # data_config = 'recbole_cdr/properties/dataset/DoubanMovie_DoubanBook_commonUser_10-core.yaml'
    data_config = 'recbole_cdr/properties/dataset/DoubanBook_DoubanMovie_commonUser_10-core.yaml'

    model_config = 'recbole_cdr/properties/model/' + model_Name + '.yaml'

    config_file_list = [overall_config, data_config, model_config]
    run_recbole_cdr(model=args.model, config_file_list=config_file_list)
