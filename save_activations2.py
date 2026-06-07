import os
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from datasets.activations import ActivationsDataset
from dictionary_learning.trainers.top_k import AutoEncoderTopK

def get_args_parser():
    parser = argparse.ArgumentParser(description="Activations extraction")
    parser.add_argument("--vpr_descriptors_dir", required=True, type=str, 
                        help="Descriptors VPR")
    parser.add_argument("--sae_checkpoint", required=True, type=str, 
                        help="older dei pesi del SAE .pt")
    parser.add_argument("--output_dir", required=True, type=str, 
                        help="final activations ")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4096)
    

    return parser

@torch.no_grad()
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo il dispositivo: {device}")

    print(f"Caricamento dei descrittori da: {args.vpr_descriptors_dir}")
    vpr_dataset = ActivationsDataset(args.vpr_descriptors_dir, device=device)
    vpr_dataloader = DataLoader(vpr_dataset, batch_size=args.batch_size, shuffle=False)
    
    sae = AutoEncoderTopK.from_pretrained(args.sae_checkpoint, k=args.k, device=device)
    sae.eval()

    
  
    state_dict = torch.load(args.sae_checkpoint, map_location=device)
    sae.load_state_dict(state_dict)
    sae.eval() 

  
    print("Estrazione delle attivazioni sparse")
    all_sparse_activations = []

    for batch in vpr_dataloader:
        batch = batch.to(device)
        

        sparse_acts = sae.encode(batch)

        # Spostiamo su CPU per salvare la memoria della GPU ed evitare crash
        all_sparse_activations.append(sparse_acts.cpu())

    # Uniamo tutti i blocchi in un'unica matrice finale
    all_sparse_activations = torch.cat(all_sparse_activations, dim=0)
    print(f"→ Matrice delle attivazioni generata! Forma: {all_sparse_activations.shape}")

    # 4. Salvataggio unico e pulito nel formato richiesto da metric.py
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(all_sparse_activations, out_path / "0.pt")
    print(f"✓ File salvato correttamente in: {out_path / '0.pt'}\n")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)