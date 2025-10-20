class BaseConfig:
    def __init__(self):
        self.domain_based = False
        
        # --- COMMONS ---
        self.parentDir  = None
        self.resultsDir = r"..\PneunomiaSeg\Segmentation\results"
        
        self.all_labels  = []
        self.only_labels = []
        
        # Paths 
        self.trainImagesDir = []
        self.trainMasksDir  = []