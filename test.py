import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from models.network import Model


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.ToTensor(),
                           normalize])
def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def main():
    parser = argparse.ArgumentParser(description='Efficient Style-Corpus Constrained Learning for Photorealistic Style Transfer')
    parser.add_argument('--content', '-c', type=str, default="images/content/56.png",
                        help='Content image path e.g. content.jpg')
    parser.add_argument('--style', '-s', type=str, default="images/style/56.png",
                        help='Style image path e.g. image.jpg')
    parser.add_argument('--output_name', '-o', type=str, default=None,
                        help='Output path for generated image, no need to add ext, e.g. out')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--model_state_path', type=str, default='models/checkpoint/model.pth',
                        help='directory for checkpoint')
    args = parser.parse_args()

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device("cuda:{}".format(args.gpu))
        print("# CUDA available:{}".format(torch.cuda.get_device_name(0)))
    else:
        device = 'cpu'

    #set model
    model = Model()
    if args.model_state_path is not None:
        model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
    model.to(device)
    c = Image.open(args.content).convert('RGB')
    s = Image.open(args.style).convert('RGB')

    c_tensor = trans(c).unsqueeze(0).to(device)
    s_tensor = trans(s).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(c_tensor, s_tensor)
    out_denorm = denorm(out, device)

    if args.output_name is None:
        c_name = os.path.splitext(os.path.basename(args.content))[0]
        s_name = os.path.splitext(os.path.basename(args.style))[0]
        args.output_name ="{}_{}".format(c_name,s_name)

    save_image(out_denorm, 'result/{}.png'.format(args.output_name), nrow=1)



if __name__ == '__main__':
    main()
