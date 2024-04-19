from torch import nn

class BackPainNN(nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        
        super().__init__()
        
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_out = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm_1 = nn.BatchNorm1d(64)
        self.batchnorm_2 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        
        x = self.relu(self.layer_1(x))
        x = self.batchnorm_1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm_2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x