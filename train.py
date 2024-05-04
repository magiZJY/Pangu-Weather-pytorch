from model import *


def train():
    model = PanguModel()
    optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=3e-6)
    epochs = 100
    for i in range(epochs):  # Loop from 1979 to 2017
        dataset_length = 5
        for step in range(dataset_length):

            inputs, inputs_surface, targets, targets_surface = LoadData(step)
            # inputs, inputs_surface, targets, targets_surface = inputs.to(device), inputs_surface.to(device), targets.to(device), targets_surface.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            # Forward pass
            outputs, outputs_surface = model(inputs, inputs_surface)

            # Calculate loss: MAE loss for both output and surface output, with additional weight for the surface loss
            # loss = TensorAbs(output-target) + TensorAbs(output_surface-target_surface) * 0.25
            loss = F.l1_loss(outputs, targets) + 0.25 * F.l1_loss(outputs_surface, targets_surface)
            total_loss += loss.item()

            # Backward pass + optimize
            loss.backward()
            optimizer.step()
            print("what???")

        print(f'Epoch [{i+1}/{epochs}], Loss: {total_loss/len(dataset_length)}')
    torch.save(model.state_dict(), 'pangu_weather_model.pth')

