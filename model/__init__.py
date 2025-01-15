def define_model(net_type='resnet18', size=224):

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

    # 根据 net_type 创建对应的模型
    if net_type == 'resnet':
        model = RN.ResNet("imagenet",
                          depth,
                          1000,
                          norm_type="batch",
                          size=size,
                          nch=3)
    elif net_type == 'resnet_ap':
        model = RNAP.ResNetAP("imagenet",
                              depth,
                              1000,
                              width=1.0,
                              norm_type="batch",
                              size=size,
                              nch=3)
    else:
        raise Exception('Unknown network architecture: {}'.format(net_type))

    return model