import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def train_and_export():
    # 1. Detección de dispositivo (CUDA / CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo seleccionado: {device}")

    # 2. Definición de rutas
    data_dir = "gold_cellular_patches"
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "resnet50_model.onnx")

    # 3. Transformaciones de imágenes
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 4. Cargar Dataset de muestra local
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"No se encontró el directorio de datos: {data_dir}")

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    num_classes = len(dataset.classes)
    print(f"[INFO] Dataset cargado desde '{data_dir}' con {len(dataset)} imágenes y {num_classes} clase(s): {dataset.classes}")

    # 5. Cargar ResNet50 y adaptar la última capa FC
    try:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    except AttributeError:
        model = models.resnet50(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # 6. Configurar Pérdida y Optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 7. Entrenamiento de Prueba (2 Épocas)
    epochs = 2
    model.train()
    print(f"\n--- Iniciando entrenamiento ResNet50 ({epochs} épocas) ---")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = (correct / total) * 100.0 if total > 0 else 0.0
        print(f"Época [{epoch}/{epochs}] - Loss: {epoch_loss:.4f} | Precisión: {epoch_acc:.2f}%")

    print("--- Entrenamiento finalizado con éxito ---\n")

    # 8. Exportar a ONNX
    print(f"[INFO] Exportando modelo a formato ONNX en '{onnx_path}'...")
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False
    )

    print(f"[ÉXITO] Modelo ResNet50 exportado en: {onnx_path}\n")

if __name__ == "__main__":
    train_and_export()
