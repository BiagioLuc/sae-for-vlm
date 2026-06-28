import torch
import torch.nn.functional as F
import tqdm
import random
import os
from PIL import Image
from utils import get_model, get_text_model
from torchvision import transforms
import argparse
from torch.utils.data import Dataset, DataLoader

def get_args_parser():
    parser = argparse.ArgumentParser("Compute embeddings", add_help=False)
    parser.add_argument("--data_path", type=str, required=True, help="Percorso alla cartella dello split (es. /content/dataset_split/val)")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_subdir", type=str, default="embeddings")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


class CityImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            
            return torch.zeros(3, 224, 224)

def load_images_and_compute_embeddings(args, model):
    image_embeddings = []
    ordered_image_paths = []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Il percorso {args.data_path} non esiste.")
        
    citta_dirs = sorted([d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))])
    print(f"-> Sottocartelle (città) trovate: {citta_dirs}")
    
    for citta in citta_dirs:
        cartella_citta = os.path.join(args.data_path, citta)
        file_ordinati = sorted(os.listdir(cartella_citta))
        for filename in file_ordinati:
            if filename.lower().endswith(("png", "jpg", "jpeg")):
                ordered_image_paths.append(os.path.join(cartella_citta, filename))

    print(f"-> Totale immagini individuate: {len(ordered_image_paths)}")
    if len(ordered_image_paths) == 0:
        print("[ATTENZIONE] Nessuna immagine trovata. Controlla la struttura delle cartelle.")
        return None

    device = args.device
    print(f"-> Spostamento del modello su device: {device}")
    if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
        model.model = model.model.to(device).eval()
    else:
        model = model.to(device).eval()
    
    # 2. Creazione del Dataset e del DataLoader parallelizzato
    dataset = CityImageDataset(ordered_image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    # Il ciclo ora itera per Batch, non per singola immagine
    for batch_tensors in tqdm.tqdm(dataloader, desc="Processing images in batches"):
        batch_tensors = batch_tensors.to(device, non_blocking=True)
        
        with torch.no_grad():
            if hasattr(model, 'encode'):
                embedding = model.encode(batch_tensors)
            else:
                embedding = model(batch_tensors)
            
            image_embeddings.append(embedding.cpu())

    if not image_embeddings:
        print("Nessuna immagine processata.")
        return None

    final_embeddings = torch.cat(image_embeddings, dim=0)        
    os.makedirs(args.output_subdir, exist_ok=True)
    embedding_output_path = os.path.join(args.output_subdir, f"embeddings_{args.model_name}.pt")
    torch.save(final_embeddings, embedding_output_path)
    print(f"-> Completato! File salvato in: {embedding_output_path}")

    return final_embeddings


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    
    print("-> Inizializzazione del modello...")
    model, processor = get_model(args) 
    
    print("-> Avvio del calcolo degli embeddings...")
    load_images_and_compute_embeddings(args, model)