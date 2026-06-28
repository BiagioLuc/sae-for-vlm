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
    print(f"Using device: {device}")
    
    #Instead of loading an image dataset (e.g., ImageFolder) 
    # and using a vision processor, we now directly load the pre-computed 
    # base model activations (descriptors) using a dedicated ActivationsDataset.
    
    print(f"Loading descriptors from: {args.vpr_descriptors_dir}")
    vpr_dataset = ActivationsDataset(args.vpr_descriptors_dir, device=device)
    vpr_dataloader = DataLoader(vpr_dataset, batch_size=args.batch_size, shuffle=False)

    # We no longer load the big vision model (like CLIP).
    # Since we already have the features ready, we just load the SAE by itself 
    # as a standalone model to process our data directly.
    
    sae = AutoEncoderTopK.from_pretrained(args.sae_checkpoint, k=args.k, device=device)
    sae.eval()

    
  
    state_dict = torch.load(args.sae_checkpoint, map_location=device)
    sae.load_state_dict(state_dict)
    sae.eval() 

  
    print("Extracting sparse activations")
    all_sparse_activations = []

    for batch in vpr_dataloader:
        batch = batch.to(device)
        
    #We no longer call `model.encode(image)` to capture 
        # hooks. We feed the raw activations batch directly into the SAE encoder.
        sparse_acts = sae.encode(batch)

        # Move tensors to CPU to conserve GPU memory and avoid Out-Of-Memory (OOM) crashes
        all_sparse_activations.append(sparse_acts.cpu())

    # Concatenate all processed batches into a single final matrix
    all_sparse_activations = torch.cat(all_sparse_activations, dim=0)
    print(f"→ Activation matrix generated! Shape: {all_sparse_activations.shape}")

    # Save everything cleanly into a single file as required by metric.pyy
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    #Script 1 chunked and saved files every N samples. 
    # Here we consolidate everything into one file named "0.pt"
    
    torch.save(all_sparse_activations, out_path / "0.pt")
    print(f"✓ File successfully saved to: {out_path / '0.pt'}\n")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
