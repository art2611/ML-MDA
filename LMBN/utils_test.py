import torch




def utils_test_LMBN(input1, device, net):
    input_img = input1.to(device)
    outputs = net(input_img)

    f1 = outputs.data.cpu()
    # flip
    input1 = input1.index_select(
        3, torch.arange(input1.size(3) - 1, -1, -1).to(device))
    input_img = input1.to(device)
    outputs = net(input_img)
    f2 = outputs.data.cpu()

    ff = f1 + f2
    if ff.dim() == 3:
        fnorm = torch.norm(
            ff, p=2, dim=1, keepdim=True)  # * np.sqrt(ff.shape[2])
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)

    else:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
    return ff