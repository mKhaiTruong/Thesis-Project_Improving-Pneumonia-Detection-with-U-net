class BaseConfig:
    def __init__(self):
        self.domain_based = False
        
        # --- COMMONS ---
        self.parentDir  = None
        self.resultsDir = r"D:\Deep_Learning_Object_Detection\randProjects\Caner_Detect\Segmentation\results"
        
        self.all_labels  = []
        self.only_labels = []
        
        # Paths 
        self.trainImagesDir = []
        self.trainMasksDir  = []