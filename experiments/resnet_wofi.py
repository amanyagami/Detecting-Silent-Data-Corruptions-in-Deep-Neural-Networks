# Load pre-trained ResNet-18 model
from src.models.resnet import resnet18
net = resnet18(pretrained=True, progress=True)
num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, 10)  # Modify the last layer to output 10 classes
net = net.to(device)
criterion = nn.CrossEntropyLoss()
net.eval()
# Testing function
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'\nTest set: Average loss: {test_loss/len(testloader):.4f}, Accuracy: {correct}/{total} ({100.*correct/total:.2f}%)\n')

#test(0)

print("Done")