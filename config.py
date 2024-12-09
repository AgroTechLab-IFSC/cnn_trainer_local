## @file config.py
#  @author Robson Costa (<mailto:robson.costa@ifsc.edu.br>)
#  @brief Configuration class.
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
import sys
import logging
from ruamel.yaml import YAML

## Configuration class.
#  @brief Get and validate configuration parameters from a configuration file based on YAML.
class Config:
    
    ## @fn __init__
    #  @brief The Config class initializer.
    #  @param _cfgFile Configuration file path.
    def __init__(self, _cfgFile):
        
        ## Configuration file
        self.configFile = _cfgFile

        ## A tree with configuration project
        self.tree = self._readConfigFile()

        # Getting CNN_LOCAL session
        logging.info("Getting 'cnn_local' key information")
        if "cnn_local" not in self.tree:
            logging.error("Not 'cnn_local' key found on configuration file")
            sys.exit("ERROR: Not 'cnn_local' key found on configuration file!!!")            
        
        ## CPU used
        self.cpuUsed = self.tree["cnn_local"]["cpu_used"]
        logging.info("Getting 'cpu_used': %d", self.cpuUsed)

        ## Train path
        self.trainPath = self.tree["cnn_local"]["train_path"]
        logging.info("Getting 'train_path': %s", self.trainPath)
        
        ## Test path
        self.testPath = self.tree["cnn_local"]["test_path"]
        logging.info("Getting 'test_path': %s", self.testPath)

        ## Validation path
        self.validationPath = self.tree["cnn_local"]["val_path"]
        logging.info("Getting 'val_path': %s", self.validationPath)

        ## Models path
        self.modelsPath = self.tree["cnn_local"]["models_path"]
        logging.info("Getting 'models_path': %s", self.modelsPath)

        ## Transforms height
        self.transformsHeight = self.tree["cnn_local"]["transforms_height"]
        logging.info("Getting 'transforms_height': %d", self.transformsHeight)

        ## Transforms width
        self.transformsWidth = self.tree["cnn_local"]["transforms_width"]
        logging.info("Getting 'transforms_width': %d", self.transformsWidth)

        ## Replications
        self.replications = self.tree["cnn_local"]["replications"]
        logging.info("Getting 'replications': %d", self.replications)

        ## Batch size
        self.batchSize = self.tree["cnn_local"]["batch_size"]
        logging.info("Getting 'batch_size': %d", self.batchSize)

        ## Model names
        self.modelNames = self.tree["cnn_local"]["model_names"]
        logging.info("Getting 'model_names': %s", self.modelNames)

        ## Epochs
        self.epochs = self.tree["cnn_local"]["epochs"]
        logging.info("Getting 'epochs': %s", self.epochs)

        ## Learning rates
        self.learningRates = self.tree["cnn_local"]["learning_rates"]
        logging.info("Getting 'learning_rates': %s", self.learningRates)

        ## Weight decays
        self.weightDecays = self.tree["cnn_local"]["weight_decays"]
        logging.info("Getting 'weight_decays': %s", self.weightDecays)
    

    ## @fn _readConfigFile
    #  @brief Read the configuration file.
    def _readConfigFile(self):
        try:
            with open(self.configFile, 'r') as _f:
                yaml = YAML(typ='safe')
                tree = yaml.load(_f)
                return tree
        except FileNotFoundError:
            logging.error("Configuration file not found!!!")
            sys.exit("ERROR: Configuration file not found!!!")