import numpy as np

class ProportionalBatchSampler:
    def __init__(self, dataset, batch_size, proportions):
        self.dataset = dataset
        self.batch_size = batch_size
        self.proportions = proportions
        self.Y_set = list(set(np.array(dataset.Y)))
        self.A_set = list(set(np.array(dataset.A)))
        self.Y_set_len = len(self.Y_set) 
        self.A_set_len = len(self.A_set) 
        self.n_classes = self.Y_set_len * self.A_set_len
        self.label_to_indices = { a*self.Y_set_len+y : np.where((np.array(self.dataset.Y) == y) & 
                                                                (np.array(self.dataset.A) == a))[0]
                                 for a in self.A_set for y in self.Y_set }
        self.labels_set = [None] * self.n_classes
        self.n_samples  = [None] * self.n_classes
        total = 0
        for a in self.A_set:
            for y in self.Y_set:
                l_inx = a*self.Y_set_len+y
                self.labels_set[l_inx] = l_inx
                self.n_samples[l_inx] = int(np.round(self.proportions[l_inx] * self.batch_size))
                total += self.n_samples[l_inx]
        #if(total < self.batch_size ):
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0  
        
    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                startI = self.used_label_indices_count[class_]
                endI = self.used_label_indices_count[class_] + self.n_samples[class_]
                indices.extend(self.label_to_indices[class_][startI:endI])
                
                self.used_label_indices_count[class_] += self.n_samples[class_]
                if self.used_label_indices_count[class_] + self.n_samples[class_] > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
                    
                self.count += self.n_samples[class_]
            
            yield indices
            
    def __len__(self):
        return len(self.data) // self.batch_size