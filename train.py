import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from collections import OrderedDict
import os 
import argparse


def main():
    parser = argparse.ArgumentParser();
    parser.add_argument('data_dir',
                    help='Folder with data')
    parser.add_argument('--save_dir', dest='save_dir', default='./',
                    help='Save directory (default: ./)')
    parser.add_argument('--arch', dest='arch', default='vgg16',
                    help='Model architecture (default: vgg16)')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.0003,
                    help='Model learning rate (default: 0.0003)')
    parser.add_argument('--epochs', dest='epochs', default=1,
                    help='Model epochs (default: 1)')
    parser.add_argument('--gpu', dest='gpu', default=False,
                    help='Model set gpu on (default: False)')
    parser.add_argument('--hidden_units', dest='hidden_units', default=25088,
                    help='Model number of nodes in the hidden layer (default: 25088)')
    
    args = parser.parse_args()
    gpu = args.gpu if hasattr(args, 'gpu') else False
    arch = args.arch if hasattr(args, 'arch') else 'vgg16'
    epochs = int(args.epochs) if hasattr(args, 'epochs') else 1
    learning_rate = float(args.learning_rate) if hasattr(args, 'learning_rate') else 0.0003
    save_dir = args.save_dir if hasattr(args, 'save_dir') else './'
    hidden_units = int(args.hidden_units) if hasattr(args, 'hidden_units') else 25088
    
    datasets = get_datasets(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if gpu else "cpu"
    model = train(datasets, device, arch = arch, epochs = epochs, lr = learning_rate, hidden_units = hidden_units)
    test_model(model, datasets['testing_loader'], device)
    save_model(model, datasets['training_dataset'], arch = arch, filename = save_dir + '/checkpoint.pth')
    print('Done')
    
def get_datasets(data_dir = 'flowers'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True)
    testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=64)
    
    return {
        'training_loader': training_loader,
        'validation_loader': validation_loader,
        'testing_loader': testing_loader,
        'training_dataset': training_dataset
    }
    
def train(datasets, device, arch = 'vgg16', epochs = 1, lr = 0.0003, hidden_units = 25088):
    model = getattr(models, arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(hidden_units, 400)),
                    ('relu', nn.ReLU()),
                    ('dropout', nn.Dropout(p=0.005)),
                    ('fc2', nn.Linear(400, 102)),
                    ('output', nn.LogSoftmax(dim=1))
                 ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    model.to(device)
    
    steps = 0
    running_loss = 0
    print_every = 5

    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in datasets['training_loader']:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in datasets['testing_loader']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(datasets['testing_loader']):.3f}.. "
                      f"Test accuracy: {accuracy/len(datasets['testing_loader']):.3f}")


                train_losses.append(running_loss/len(datasets['training_loader']))
                test_losses.append(test_loss/len(datasets['testing_loader']))
                running_loss = 0
                model.train()
    return model

def test_model(model, testing_loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testing_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test accuracy of model: {round(100 * correct / total, 2)}%")
    
def save_model(model, training_dataset, arch = 'vgg16', filename = 'checkpoint.pth'):    
    model.cpu()
    checkpoint = {
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': training_dataset.class_to_idx,
        'model_arch': arch
    }

    torch.save(checkpoint, filename)
    
if __name__ == '__main__':
    main()
    