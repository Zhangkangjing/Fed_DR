import warnings
from collections import OrderedDict
import argparse
import flwr as fl
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

from datasets import prepare_dataset


warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 50

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
        "--num_samples",
        type=int,
        default=1300,

        help="Number of traindataset of federated learning ",
    )
parser.add_argument(
        "--trainset_1_num",
        type=int,
        default=500,

        help="Number of trainset_1_num of federated learning ",
    )
parser.add_argument(
    "--server_address",
    type=str,
    default="192.168.0.102:8080",
    help=f"gRPC server address (deafault '192.168.0.100:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)


def net():
    net = models.squeezenet1_1(pretrained=True)
    net.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1, 1), stride=(1, 1))
    net.num_classes = 5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    return net

def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()  # Set the model to training mode
        running_loss = 0.0
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            running_loss += loss.item()
            # Print average loss for the epoch
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.2f}")


def test(net, testloader, device):
    args = parser.parse_args()
    """Validate the model on the test set."""
    net.eval()
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    print(f"{args.cid}Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return loss, accuracy

class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that trains a MobileNetV3 model for CIFAR-10 or a much smaller CNN
    for MNIST."""

    def __init__(self, trainloader, valloader) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = net()
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        # Read hyperparameters from config set by the server
        batch, epochs = config["batch_size"], config["epochs"]
        # Construct dataloader
        trainloader = self.trainloader
        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # Train
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)
        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        # Construct dataloader
        valloader = self.valloader
        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device)
        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}



def main():
    args = parser.parse_args()
    print(args)
    assert args.cid < NUM_CLIENTS
    trainloaders, valloaders, testloader = prepare_dataset(num_samples=args.num_samples, trainset_1_num=args.trainset_1_num, batch_size=32)

    # Start Flower client setting its associated data partition
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainloader=trainloaders[args.cid], valloader=valloaders[args.cid]
        ).to_client()
    )


if __name__ == "__main__":
    main()