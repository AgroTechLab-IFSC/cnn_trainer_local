import os
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import config as config
from cnn import CNN
import torch
from torchvision import datasets
from torchvision.transforms import v2

## Configuration file name
CONFIG_FILE = 'cnn_trainer.yml'
LOG_FILE = 'cnn_trainer.log'

def define_transforms(height, width):
    """
    Define transforms for the images.

    Parameters:
        height (int): Images height.
        width (int): Images width.

    Returns:
        (dict): Data transforms.
    """   
    logging.info("Defining transforms")
    print("\tDefining transforms...", flush=True)
    data_transforms = {
        'train' : v2.Compose([
                    v2.Resize((height, width)),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test'  : v2.Compose([
                    v2.Resize((height, width)),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'validation' : v2.Compose([
                    v2.Resize((height, width)),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }
    return data_transforms

def read_images(data_transforms, train_path, val_path, test_path):
    """
    Read images (train, validation and test) from their respective directories.

    Parameters:
        data_transforms (dict): Tranforms to be applied to the images.
        train_path (str): Path to train images directory.
        val_path (str): Path to validation images directory.
        test_path (str): Path to test images directory.

    Returns:
        (dict): A dict mapping keys to the:
                * 'train_data': (datasets.ImageFolder) Train data.
                * 'validation_data': (datasets.ImageFolder) Validation data.
                * 'test_data': (datasets.ImageFolder) Test data.
    """
    logging.info("Reading images")
    print("\tReading images...", flush=True)

    ## Train data
    train_data = datasets.ImageFolder(train_path, transform=data_transforms['train'])

    ## Validation data
    validation_data = datasets.ImageFolder(val_path, transform=data_transforms['validation'])

    ## Test data
    test_data = datasets.ImageFolder(test_path, transform=data_transforms['test'])
    
    return train_data, validation_data, test_data

def main():
    """Main function."""
    logging.info("Starting CNN Trainer (Local)")
    print("Starting CNN Trainer (Local)", flush=True)

    # Check CPU count
    cpu_count = multiprocessing.cpu_count()
    logging.info(f"CPU count: {cpu_count}")
    print(f"\tCPU count: {cpu_count}", flush=True)

    # Read configuration file
    logging.info(f"Reading configuration file {CONFIG_FILE}")
    print("\tReading configuration file... ", end='', flush=True)
    cfgObj = config.Config(cfgFile=CONFIG_FILE)    
    print("[OK]", flush=True)

    # Validate CPU used
    if cfgObj.cpuUsed > cpu_count:
        cfgObj.cpuUsed = cpu_count
        logging.warning(f"Parameter 'cpu_used' is greater than available CPUs. Using all {cpu_count} available CPUs")
        print(f"\t\tParameter 'cpu_used' is greater than available CPUs. Using all {cpu_count} available CPUs...", flush=True)

    # Read data and create CNN object
    data_transforms = define_transforms(cfgObj.transformsHeight, cfgObj.transformsWidth)
    train_data, validation_data, test_data = read_images(data_transforms, cfgObj.trainPath, cfgObj.validationPath, cfgObj.testPath)
    cnn = CNN(train_data, validation_data, test_data, cfgObj.batchSize)
    
    # Create an executor to control the number of subprocesses
    process_queue = ProcessPoolExecutor(max_workers = cfgObj.cpuUsed)

    # Create subprocesses
    proc_num = cfgObj.modelNames.__len__() * cfgObj.epochs.__len__() * cfgObj.learningRates.__len__() * cfgObj.weightDecays.__len__()
    logging.info(f"Creating {proc_num} subprocesses that will generate {(proc_num * cfgObj.replications)} models")
    print(f"\tCreating {proc_num} subprocesses that will generate {(proc_num * cfgObj.replications)} models", flush=True)
    generalBegin = time.time()
    futures = []
    for model_name in cfgObj.modelNames:
        for epoch in cfgObj.epochs:
            for learning_rate in cfgObj.learningRates:
                for weight_decay in cfgObj.weightDecays:
                    futures.append(process_queue.submit(cnn.create_and_train_cnn, model_name, epoch, learning_rate, weight_decay, cfgObj.replications))

    # Wait for results
    logging.info(f"Waiting results of {len(futures)} subprocesses")
    print(f"\tWaiting results of {len(futures)} subprocesses...", flush=True)
    for future in as_completed(futures):
        result_name, acc_avg, iter_acc_max, duration = future.result()
        logging.info(f"{result_name} (Avg. accuracy: {acc_avg:.4f} - Best replication: {iter_acc_max} - Duration: {round(duration, 2):.2f})")
        print(f"\t\t{result_name} (Avg. accuracy: {acc_avg:.4f} - Best replication: {iter_acc_max} - Duration: {round(duration, 2):.2f})", flush=True)
    
    # Compute total time
    generalEnd = time.time()
    totalDuration = round((generalEnd - generalBegin), 2)
    logging.info(f"Total duration: {totalDuration:.2f} seconds")
    print(f"\tTotal duration: {totalDuration:.2f} seconds", flush=True)
    
if __name__ == '__main__':
    """Main function (entry point)."""
    # Setup logging
    logging.basicConfig(filename=LOG_FILE, format='%(asctime)s %(levelname)s - %(message)s', encoding='utf-8', level=logging.INFO)
    
    # Call main function
    main()