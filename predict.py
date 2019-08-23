import torch
from torchvision import transforms, models
import argparse
from PIL import Image
import numpy as np
import json


def main():
    parser = argparse.ArgumentParser();
    parser.add_argument('image_path',
                    help='Path to image')
    parser.add_argument('checkpoint_path',
                    help='Checkpoint path')
    parser.add_argument('--top_k', dest='top_k', default='5',
                    help='Count of top (default: 5)')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json',
                    help='Path to file with real category names (default: cat_to_name.json)')
    parser.add_argument('--gpu', dest='gpu', default=False,
                    help='Model set gpu on (default: False)')
    
    args = parser.parse_args()
    image_path = args.image_path
    checkpoint_path = args.checkpoint_path
    top_k = int(args.top_k) if hasattr(args, 'top_k') else 5
    category_names = args.category_names if hasattr(args, 'category_names') else 'cat_to_name.json'
    gpu = args.gpu if hasattr(args, 'gpu') else False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if gpu else "cpu"
    
    model_loaded = load_checkpoint(checkpoint_path)
    probs, classes = predict(image_path, model_loaded, device = device, topk = top_k)
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    for cl, prob in zip(classes, probs):
        print(cat_to_name[str(cl)], f"{round(prob * 100, 2)}%" )

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = getattr(models, checkpoint['model_arch'])(pretrained=True)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.classifier = checkpoint['classifier'] 
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    image = Image.open(image)
    image_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    image = np.array(image_transforms(image))
    image = torch.from_numpy(image).float()
    
    return image

def predict(image_path, model, device = 'cpu', topk = 5):
    model.eval()
    model.to(device)
    
    image = process_image(image_path);
    image = image.unsqueeze_(0)

    output = model.forward(image.to(device))
    output = torch.exp(output)
    
    probs, classes = output.topk(topk, dim=1)
    return probs[0].detach().cpu().numpy(), classes[0].cpu().numpy()

if __name__ == '__main__':
    main()