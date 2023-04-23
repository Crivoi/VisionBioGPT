import urllib.request
import skimage
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms
import torchxrayvision as xrv

if __name__ == '__main__':
    model_name = "densenet121-res224-chex"

    img_url = "https://huggingface.co/spaces/torchxrayvision/torchxrayvision-classifier/resolve/main/16747_3_1.jpg"
    img_path = "xray.jpg"
    urllib.request.urlretrieve(img_url, img_path)

    model = xrv.models.get_model(model_name)

    img = skimage.io.imread(img_path)
    img = xrv.datasets.normalize(img, 255)

    skimage.io.imshow(img)

    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    # Add color channel
    img = img[None, :, :]

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

    img = transform(img)

    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        preds = model(img).cpu()
        output = {
            k: float(v)
            for k, v in zip(xrv.datasets.default_pathologies, preds[0].detach().numpy())
        }
    print(output)
