import sys
import logging
from ruamel.yaml import YAML

class Config:
    """
    Get and validate configuration parameters from a configuration file based on YAML. These parameters will be saved as class attributes.
    
    Parameters:
        cfgFile (str): Configuration file path.

    Attributes:
        cpuUsed (int): Number of CPUs to be used.
        trainPath (str): Directory path of train data.
        testPath (str): Directory path of test data.
        validationPath (str): Directory path of validation data.
        modelsPath (str): Directory path of models that will be generated.
        transformsHeight (int): Images transforms height.
        transformsWidth (int): Images transforms width.
        replications (int): Number of replications used at each trained model.
        batchSize (int): Batch size.
        modelNames (list): List of model names to be trained.
        epochs (list): List of epochs to be trained.
        learningRates (list): List of learning rates to be trained.
        weightDecays (list): List of weight decays to be used at train.
        tree (dict): Configuration file tree (from YAML object).
    """
    
    def __init__(self, cfgFile):
        """The Config class constructor."""
       
        self.tree = self.readConfigFile(cfgFile)

        # Getting CNN_LOCAL session
        logging.info("Getting 'local' key information")
        if "local" not in self.tree:
            logging.error("Not 'local' key found on configuration file")
            sys.exit("ERROR: Not 'local' key found on configuration file!!!")            
        
        self.cpuUsed = self.tree["local"]["cpu_used"]
        logging.info("Getting 'cpu_used': %d", self.cpuUsed)

        self.trainPath = self.tree["local"]["train_path"]
        logging.info("Getting 'train_path': %s", self.trainPath)
        
        self.testPath = self.tree["local"]["test_path"]
        logging.info("Getting 'test_path': %s", self.testPath)

        self.validationPath = self.tree["local"]["val_path"]
        logging.info("Getting 'val_path': %s", self.validationPath)

        self.modelsPath = self.tree["local"]["models_path"]
        logging.info("Getting 'models_path': %s", self.modelsPath)

        self.transformsHeight = self.tree["local"]["transforms_height"]
        logging.info("Getting 'transforms_height': %d", self.transformsHeight)

        self.transformsWidth = self.tree["local"]["transforms_width"]
        logging.info("Getting 'transforms_width': %d", self.transformsWidth)

        self.replications = self.tree["local"]["replications"]
        logging.info("Getting 'replications': %d", self.replications)

        self.batchSize = self.tree["local"]["batch_size"]
        logging.info("Getting 'batch_size': %d", self.batchSize)

        self.modelNames = self.tree["local"]["model_names"]
        logging.info("Getting 'model_names': %s", self.modelNames)

        self.epochs = self.tree["local"]["epochs"]
        logging.info("Getting 'epochs': %s", self.epochs)

        self.learningRates = self.tree["local"]["learning_rates"]
        logging.info("Getting 'learning_rates': %s", self.learningRates)

        self.weightDecays = self.tree["local"]["weight_decays"]
        logging.info("Getting 'weight_decays': %s", self.weightDecays)
    
    def readConfigFile(self, configFile):
        """
        Read the configuration file.
        
        Parameters:
            configFile (str): Configuration file path.
        
        Returns:
            (dict): Configuration tree (from YAML object).
        """
        try:
            with open(self.configFile, 'r') as _f:
                yaml = YAML(typ='safe')
                tree = yaml.load(_f)
                return tree
        except FileNotFoundError:
            logging.error("Configuration file not found!!!")
            sys.exit("ERROR: Configuration file not found!!!")