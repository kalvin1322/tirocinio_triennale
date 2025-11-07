import torch
import torch.nn as nn

class PostProcessNet(nn.Module):
    """
    Una piccola rete neurale per il post-processing (es. raffinamento, 
    denoising, rimozione artefatti) dell'output di un'altra rete.

    Args:
        in_channels (int): Canali dell'immagine/feature map in input
                           (es. 1 per maschera, 3 per RGB).
        out_channels (int): Canali desiderati in output.
        hidden_channels (int): Numero di canali negli strati intermedi.
                               Controlla la "dimensione" della rete.
        use_residual (bool): Se True, aggiunge l'input all'output (impara
                             la "correzione"). Richiede che in_channels 
                             e out_channels siano compatibili.
    """
    def __init__(self, in_channels, out_channels, hidden_channels=32, use_residual=True):
        super(PostProcessNet, self).__init__()
        
        self.use_residual = use_residual

        # Il "blocco" principale che elabora l'immagine
        self.refiner = nn.Sequential(
            # Mantiene le dimensioni HxW grazie a padding=1
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Strato intermedio
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Strato finale: proietta sui canali di output
            # kernel_size=1 è efficiente e agisce come un per-pixel fully-connected
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )

        # Gestione della connessione residua (shortcut)
        if self.use_residual:
            if in_channels == out_channels:
                # Se i canali corrispondono, la shortcut è l'identità
                self.shortcut = nn.Identity()
            else:
                # Se i canali non corrispondono, usiamo una Conv 1x1
                # per "adattare" l'input da sommare
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Nota: non applichiamo un'attivazione finale (es. Sigmoid o Tanh)
        # Questo la rende flessibile. L'attivazione andrebbe applicata
        # *dopo* questa rete, a seconda del task (es. Sigmoid per maschere binarie).

    def forward(self, x):
        """
        x: Tensor in input (output della rete precedente)
           Formato: (batch_size, in_channels, height, width)
        """
        
        # Calcola l'elaborazione principale
        processed_output = self.refiner(x)

        if self.use_residual:
            # Applica la shortcut all'input originale 'x'
            shortcut_output = self.shortcut(x)
            # Somma l'input originale (adattato) all'output elaborato
            return shortcut_output + processed_output
        else:
            # Ritorna solo l'output elaborato
            return processed_output