def define_model(dataset,norm_type,net_type,nch,depth,width,nclass, size):

    if  net_type == 'resnet':
        model = RN.ResNet(dataset,
                          depth,
                          nclass,
                          norm_type=norm_type,
                          size=size,
                          nch=nch)
    elif net_type == 'resnet_ap':
        model = RNAP.ResNetAP(dataset,
                              depth,
                              nclass,
                              width=width,
                              norm_type=norm_type,
                              size=size,
                              nch=nch)
    elif net_type == 'efficient':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    elif net_type == 'densenet':
        model = DN.densenet_cifar(nclass)
    elif net_type == 'convnet':
        width = int(128 * width)
        model = CN.ConvNet(nclass,
                           net_norm=norm_type,
                           net_depth=depth,
                           net_width=width,
                           channel=nch,
                           im_size=(size, size))
    else:
        raise Exception('unknown network architecture: {}'.format(net_type))
    return model
