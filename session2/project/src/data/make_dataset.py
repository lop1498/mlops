# -*- coding: utf-8 -*-
import click
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df_train = pd.read_csv(input_filepath+'/mnist_train.csv')
    df_test = pd.read_csv(input_filepath+'/mnist_test.csv')

    scaler = StandardScaler().fit(df_test.values[:,1:])
    test_data = scaler.transform(df_test.values[:,1:])  
    
    scaler = StandardScaler().fit(df_train.values[:,1:])
    train_data = scaler.transform(df_train.values[:,1:])  
    
    np.save(output_filepath+'/train_data',  train_data)
    np.save(output_filepath+'/test_data',  test_data)
    np.save(output_filepath+'/train_labels',  np.array(df_train)[:,0])
    np.save(output_filepath+'/test_labels',  np.array(df_test)[:,0])

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
