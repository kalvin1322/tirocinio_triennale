import torch
import torch.nn as nn

class SimpleResNet(nn.Module):
    """
    Una semplice rete residua (FCN) per il post-processing.
    
    Questa rete impara una mappatura residua F(x) e restituisce x + F(x).
    È molto leggera e non utilizza né downsampling né upsampling (non è una U-Net).
    Perfetta per un rapido benchmarking.
    
    Parametri:
    - in_channels (int): Canali di input (es. 1 per un'immagine SIRT)
    - out_channels (int): Canali di output (deve essere == in_channels per la connessione residua)
    - num_layers (int): Numero di strati convoluzionali nascosti nel "corpo" della rete.
    - features (int): Numero di filtri (canali) negli strati nascosti.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_layers: int = 4, features: int = 32):
        super().__init__()
        
        if in_channels != out_channels:
            raise ValueError("in_channels e out_channels devono essere uguali per questa rete residua")

        # Blocco helper, definito nello stile del tuo esempio U-Net.
        # Questo è un blocco molto più semplice (un solo strato) per la massima leggerezza.
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU()
            )

        # --- Input ---
        # Converte l'input da in_channels a 'features'
        self.input_conv = conv_block(in_channels, features)

        # --- Corpo (Body) ---
        # Una serie di strati che lavorano tutti alla stessa risoluzione
        self.body = nn.ModuleList()
        for _ in range(num_layers):
            self.body.append(conv_block(features, features))

        # --- Output ---
        # Converte l'output da 'features' a out_channels (es. 1)
        # Questo strato NON ha ReLU, poiché produce il residuo finale.
        self.output_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Salva una copia dell'input originale per la connessione residua
        identity = x

        # 1. Strato di Input
        x = self.input_conv(x)
        
        # 2. Corpo della rete
        for block in self.body:
            x = block(x)
            
        # 3. Strato di Output (calcola il residuo)
        residual = self.output_conv(x)
        
        # 4. Aggiungi la connessione residua
        # L'output è l'Input Originale + il Residuo imparato
        out = identity + residual
        
        return out