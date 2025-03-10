import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch, rank):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0 and rank == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader, rank):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    if rank == 0:
        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} '
              f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

def main(rank, world_size):
    setup(rank, world_size)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        cleanup()
        return
    
    device = torch.device(f"cuda:{rank}")
    print(f"In {rank}, using : {device}")
    print(f"CUDA device name: {torch.cuda.get_device_name(rank)}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    model = Net().to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    
    for epoch in range(1, 3):
        train_sampler.set_epoch(epoch)
        train(model, device, train_loader, optimizer, epoch, rank)
        test(model, device, test_loader, rank)
        scheduler.step()
    
    if rank == 0:
        torch.save(model.state_dict(), "mnist_ddp.pt")
    
    cleanup()

if __name__ == "__main__":
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Find {n_gpus} CUDA devices")
        
        import torch.multiprocessing as mp
        mp.spawn(main, args=(n_gpus,), nprocs=n_gpus, join=True)
    else:
        print("CUDA not available!")
