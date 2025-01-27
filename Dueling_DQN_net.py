import torch
from torch import nn

class Dueling_DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.Linear1 = nn.Linear(input_size,32)

        self.Linear2_adv = nn.Linear(32,32)
        self.Linear2_val = nn.Linear(32,32)

        self.Linear3_adv = nn.Linear(32,output_size)
        self.Linear3_val = nn.Linear(32,1)

        # 7*32 + 32*32 + 32*32 + 32*5 + 32*1 + 32 + 32 + 32 + 5 + 1 = 2566

    def forward(self,x):
        x = (x - 1.5)/1.5

        x = self.Linear1(x)
        x = nn.functional.relu(x)

        adv = self.Linear2_adv(x)
        adv = nn.functional.relu(adv)
        adv = self.Linear3_adv(adv)

        val = self.Linear2_val(x)
        val = nn.functional.relu(val)
        val = self.Linear3_val(val)

        x = val + adv - torch.mean(adv,dim=-1,keepdim=True)

        return x
    
if __name__ == "__main__":
    import torch
    net = Dueling_DQN(7,5)
    
    x = torch.randn(2,7)
    y = net(x)