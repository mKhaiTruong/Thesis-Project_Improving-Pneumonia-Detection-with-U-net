class EarlyStopping:
    
    def __init__(self, patience=7, mode='max'):
        self.patience = patience
        self.mode     = mode
        self.counter  = 0
        
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None or \
            (self.mode == 'max' and score > self.best_score) or \
                (self.mode == 'min' and score < self.best_score):
                    self.best_score = score
                    self.counter    = 0
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True