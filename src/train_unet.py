import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------
# Definición de Arquitectura U-Net Básica
# ---------------------------------------------------------
class DoubleConv(nn.Module):
    """Bloque de doble convolución: (Conv2d -> BatchNorm2d -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """Arquitectura U-Net básica para pruebas de segmentación/procesamiento"""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up2(x)
        
        logits = self.outc(x)
        return logits

# ---------------------------------------------------------
# Función de Entrenamiento y Exportación ONNX
# ---------------------------------------------------------
def train_and_export():
    # 1. Detección de dispositivo (CUDA / CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo seleccionado: {device}")

    # 2. Definición de rutas
    data_dir = "gold_cellular_patches"
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "unet_model.onnx")

    # 3. Transformaciones de imágenes
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 4. Cargar Dataset
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"No se encontró el directorio de datos: {data_dir}")

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    print(f"[INFO] Dataset cargado desde '{data_dir}' con {len(dataset)} imágenes para U-Net.")

    # 5. Instanciar Modelo U-Net
    model = UNet(in_channels=3, out_channels=1).to(device)

    # 6. Configurar Pérdida y Optimizador
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 7. Entrenamiento de Prueba (2 Épocas)
    epochs = 2
    model.train()
    print(f"\n--- Iniciando entrenamiento U-Net ({epochs} épocas) ---")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total_samples = 0

        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            # Generar máscara objetivo sintética (1 canal) a partir de la intensidad media del parche
            target_masks = inputs.mean(dim=1, keepdim=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target_masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        print(f"Época [{epoch}/{epochs}] - Loss: {epoch_loss:.4f}")

    print("--- Entrenamiento U-Net finalizado con éxito ---\n")

    # 8. Exportar a ONNX
    print(f"[INFO] Exportando modelo U-Net a formato ONNX en '{onnx_path}'...")
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

    print(f"[ÉXITO] Modelo U-Net exportado en: {onnx_path}\n")

if __name__ == "__main__":
    train_and_export()
