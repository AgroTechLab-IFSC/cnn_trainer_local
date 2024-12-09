## @file cnn_local.py
#  @author Wilson Castello Branco Neto (<mailto:wilson.castello@ifsc.edu.br>) / Robson Costa (<mailto:robson.costa@ifsc.edu.br>)
#  @brief CNN Trainer.
#  @version 0.1.0
#  @since 06/12/2024
#  @date 09/12/2024
#  @copyright Copyright &copy; since 2024 <a href="https://agrotechlab.lages.ifsc.edu.br" target="_blank">AgroTechLab</a>.\n
#  ![LICENSE license](../figs/license.png)<br>
#  Licensed under the CC BY-NC-SA (<i>Creative Commons Attribution-NonCommercial-ShareAlike</i>) 4.0 International Unported License (the <em>"License"</em>). You may not
#  use this file except in compliance with the License. You may obtain a copy of the License <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode" target="_blank">here</a>.
#  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an <em>"as is" basis, 
#  without warranties or  conditions of any kind</em>, either express or implied. See the License for the specific language governing permissions 
#  and limitations under the License.
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import config
from cnn import CNN
import torch
from torchvision import datasets
from torchvision.transforms import v2

## Configuration file name
CONFIG_FILE = 'cnn_local.yml'

## @fn define_transforms
#  @brief Define transforms for the images
#  @param height Height
#  @param width Width
#  @return Data transforms
def define_transforms(height, width):
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

## @fn read_images
#  @brief Read images
#  @param data_transforms Data transforms
#  @param train_path Train path
#  @param val_path Validation path
#  @param test_path Test path
#  @return Train data, validation data and test data
def read_images(data_transforms, train_path, val_path, test_path):
    logging.info("Reading images")
    print("\tReading images...", flush=True)

    ## Train data
    train_data = datasets.ImageFolder(train_path, transform=data_transforms['train'])

    ## Validation data
    validation_data = datasets.ImageFolder(val_path, transform=data_transforms['validation'])

    ## Test data
    test_data = datasets.ImageFolder(test_path, transform=data_transforms['test'])
    
    return train_data, validation_data, test_data

## @fn main
#  @brief Main function
def main():
    logging.info("Starting CNN Trainer (Local)")
    print("Starting CNN Trainer (Local)", flush=True)

    # Check CPU count
    cpu_count = multiprocessing.cpu_count()
    logging.info(f"CPU count: {cpu_count}")
    print(f"\tCPU count: {cpu_count}", flush=True)

    # Read configuration file
    logging.info(f"Reading configuration file {CONFIG_FILE}")
    print("\tReading configuration file... ", end='', flush=True)
    cfgObj = config.Config(CONFIG_FILE)    
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
    
## Main function
if __name__ == '__main__':
    ## LOG file name
    filename = "cnn_local.log"
    ## LOG format
    format = '%(asctime)s %(levelname)s - %(message)s'
    ## LOG level
    level = logging.INFO

    logging.basicConfig(filename=filename, format=format, level=level)
    main()