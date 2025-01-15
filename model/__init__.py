from model.resnet_ap import ResNetAP
def define_model(net_type='resnet'):

    # 定义一个字典映射 net_type 到 depth
    resnet_depths = {
        'resnet18': 18,
        'resnet34': 34,
        'resnet50': 50,
        'resnet101': 101,
        'resnet152': 152
    }

    # 根据 net_type 获取深度
    if net_type in resnet_depths:
        depth = resnet_depths[net_type]
    else:
        raise Exception('Unknown network architecture: {}'.format(net_type))

    model = ResNetAP("imagenet",
                            depth,
                            1000,
                            width=1.0,
                            norm_type="batch",
                            size=size,
                            nch=3)

    return model