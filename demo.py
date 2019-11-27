"""
 *
 * @author Taha Emara
 * Website: http://www.emaraic.com
 * Email  : taha@emaraic.com
 * Created on: 2019-11-27
"""
import argparse
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import torch
from models import liteseg_mobilenet, liteseg_shufflenet
from dataloaders import utils as dataloaders_utils
import numpy as np


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_path', required=False,
                    help='path to input image', default='./samples/frankfurt_000000_000294_leftImg8bit.png')
    ap.add_argument('--model', required=False,
                    help='name of the backbone network',
                    default='mobilenet')
    ap.add_argument('--gpu', required=False, dest='gpu', action='store_true',
                    help='use gpu')
    ap.add_argument('--no-gpu', required=False, dest='gpu', action='store_false',
                    help='use cpu')
    ap.set_defaults(gpu=False)
    args = ap.parse_args()
    return args


def main(args):
    image_path = args.img_path
    backbone_network = args.model
    use_gpu = args.gpu



    if backbone_network == 'mobilenet':
        model = liteseg_mobilenet.RT(pretrained=False)
        model.load_state_dict(torch.load("pretrained_models/liteseg-mobilenet-cityscapes.pth", map_location='cpu'))
    else:
        model = liteseg_shufflenet.RT(pretrained=False)
        model.load_state_dict(torch.load("pretrained_models/liteseg-shufflenet-cityscapes.pth", map_location='cpu'))

    if use_gpu:
        torch.cuda.set_device(device=0)
        model.cuda()
    model.eval()
    img = Image.open(image_path)

    loader = transforms.Compose([
        transforms.Resize((1024, 2048)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    img_tensor = loader(img).unsqueeze(0)
    input_image = Variable(img_tensor, requires_grad=False)
    if use_gpu:
        input_image = input_image.cuda()

    with torch.no_grad():
        outputs = model.forward(input_image)
        predictions = torch.max(outputs, 1)[1]
        pred = predictions.detach().cpu().numpy()

    pred_color = dataloaders_utils.decode_segmap_cv(pred, 'cityscapes')
    pred_color=pred_color[...,::-1]

    input_image=np.array(img)[...,::-1]
    overlayed_img = 0.4*input_image + 0.6*pred_color
    im = Image.fromarray(overlayed_img.astype('uint8'), 'RGB')
    im.show()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
