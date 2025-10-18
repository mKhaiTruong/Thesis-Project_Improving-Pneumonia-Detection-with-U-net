import segmentation_models_pytorch as smp
import torchvision

def get_unet_model(model_name, encoder, encoder_weights='imagenet', len_classes=1, activation=None):
    
    model_name = model_name.lower()
    
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name    = encoder,
            encoder_weights = encoder_weights,
            classes         = len_classes,
            activation      = activation
        )
    
    elif model_name == 'unetpp':
        model = smp.UnetPlusPlus(
            encoder_name    = encoder,
            encoder_weights = encoder_weights,
            classes         = len_classes,
            activation      = activation
        )
    
    elif model_name == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name    = encoder,
            encoder_weights = encoder_weights,
            classes         = len_classes,
            activation      = activation
        )
    
    elif model_name == 'pspnet':
        model = smp.PSPNet(
            encoder_name    = encoder,
            encoder_weights = encoder_weights,
            classes         = len_classes,
            activation      = activation
        )
    
    elif model_name == 'fpn':
        model = smp.FPN(
            encoder_name    = encoder,
            encoder_weights = encoder_weights,
            classes         = len_classes,
            activation      = activation
        )
    
    elif model_name == 'linknet':
        model = smp.Linknet(
            encoder_name    = encoder,
            encoder_weights = encoder_weights,
            classes         = len_classes,
            activation      = activation
        )
    
    elif model_name == 'segformer':
        model = smp.Segformer(
            encoder_name    = encoder,  
            encoder_weights = encoder_weights,
            classes         = len_classes,
            activation      = activation
        )
    
    elif model_name == 'dpt':
        model = smp.DPT(
            encoder_name    = encoder, 
            encoder_weights = encoder_weights,
            classes         = len_classes,
            activation      = activation
        )
    
    elif model_name == 'fcn':
        model = torchvision.models.segmentation.fcn_resnet50(
            weights=None,
            pretrained  = (encoder_weights=='imagenet'),
            num_classes = len_classes,
        )
        
    elif model_name == 'deeplabv3':
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None,
            pretrained  = (encoder_weights=='imagenet'),
            num_classes = len_classes,
        )
        
    else:
        raise ValueError(f"Model {model_name} chưa implement. Thử các tên: unet, deeplabv3plus, pspnet, fpn, linknet, fcn, deeplabv3")

    return model