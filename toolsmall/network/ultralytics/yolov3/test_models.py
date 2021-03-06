from models import Darknet
import torch

if __name__=="__main__":
    device = torch.device("cuda")
    # Create model
    # Initialize model
    model = Darknet("cfg/yolov3-spp.cfg").to(device)
    print(model)

    x = torch.rand([1,3,416,416]).to(device)
    # model.train()
    # pred = model(x)
    """
    :pred  list[
                Tensor[1,3,13,13,85]  # stride=32
                Tensor[1,3,26,26,85], # stride=16
                Tensor[1,3,52,52,85], # stride=8
                ]
    """

    model.eval()
    pred = model(x,augment=False)[0]
    # :pred Tensor[1,10647,85]
    print()