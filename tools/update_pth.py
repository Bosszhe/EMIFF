import torch

def update_backbone(model, filename, map_locaiton='cpu'):

    # from IPython import embed
    # embed(header='update')
    filename = '/home/wangz/wangzhe21/VIMI/work_dirs/0512_VIMI/best_car_3d_0.5_epoch_10.pth'
    checkpoint = torch.load(filename)
    state_dict = checkpoint['state_dict']
    backbone_i_new = {k.replace('backbone_i.', ''): v for k, v in state_dict.items() if k.startswith('backbone_i.')}
    neck_i_new = {k.replace('neck_i.', ''): v for k, v in state_dict.items() if k.startswith('neck_i.')}
    inf_compressor_new = {k.replace('inf_compressor.', ''): v for k, v in state_dict.items() if k.startswith('inf_compressor.')}
    ms_block_inf_new = {k.replace('ms_block_inf.', ''): v for k, v in state_dict.items() if k.startswith('ms_block_inf.')}
    dcn_up_conv_i_new = {k.replace('dcn_up_conv_i.', ''): v for k, v in state_dict.items() if k.startswith('dcn_up_conv_i.')}

    model.backbone_i.load_state_dict(backbone_i_new)
    model.neck_i.load_state_dict(neck_i_new)
    model.inf_compressor.load_state_dict(inf_compressor_new)
    model.ms_block_inf.load_state_dict(ms_block_inf_new)
    model.dcn_up_conv_i.load_state_dict(dcn_up_conv_i_new)

    # neck_i_new_2 = {k.replace('neck_i.', ''): v for k, v in neck_i_new.items()}




    return model

import torch


def update_backbone_vqvae_veh(model, filename, map_locaiton='cpu'):

    # from IPython import embed
    # embed(header='update')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # filename = '/home/wangz/wangzhe21/VIMI/work_dirs/0515_VIMI_VQVAE_960x540_12e_bs2x1/best_car_3d_0.5_epoch_10.pth'
    filename = '/home/wangz/wangzhe21/VIMI/work_dirs/0711_VIMI_VQVAE_Veh_B2_960x540_12e_bs2x1/best_car_3d_0.5_epoch_11.pth'
    checkpoint = torch.load(filename, map_location = device)
    state_dict = checkpoint['state_dict']

    backbone_v_new = {k.replace('backbone_v.', ''): v for k, v in state_dict.items() if k.startswith('backbone_v.')}
    neck_v_new = {k.replace('neck_v.', ''): v for k, v in state_dict.items() if k.startswith('neck_v.')}
    dcn_up_conv_v_new = {k.replace('dcn_up_conv_v.', ''): v for k, v in state_dict.items() if k.startswith('dcn_up_conv_v.')}

    model.backbone_v.load_state_dict(backbone_v_new)
    model.neck_v.load_state_dict(neck_v_new)
    model.dcn_up_conv_v.load_state_dict(dcn_up_conv_v_new)

    return model 
    
def update_backbone_vqvae(model, filename, map_locaiton='cpu'):

    # from IPython import embed
    # embed(header='update')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # filename = '/home/wangz/wangzhe21/VIMI/work_dirs/0515_VIMI_VQVAE_960x540_12e_bs2x1/best_car_3d_0.5_epoch_10.pth'
    filename = '/home/wangz/wangzhe21/VIMI/work_dirs/0711_VIMI_VQVAE_Veh_B2_960x540_12e_bs2x1/best_car_3d_0.5_epoch_11.pth'
    checkpoint = torch.load(filename, map_location = device)
    state_dict = checkpoint['state_dict']

    backbone_v_new = {k.replace('backbone_v.', ''): v for k, v in state_dict.items() if k.startswith('backbone_v.')}
    neck_v_new = {k.replace('neck_v.', ''): v for k, v in state_dict.items() if k.startswith('neck_v.')}
    dcn_up_conv_v_new = {k.replace('dcn_up_conv_v.', ''): v for k, v in state_dict.items() if k.startswith('dcn_up_conv_v.')}

    model.backbone_v.load_state_dict(backbone_v_new)
    model.neck_v.load_state_dict(neck_v_new)
    model.dcn_up_conv_v.load_state_dict(dcn_up_conv_v_new)


    # backbone_i_new = {k.replace('backbone_i.', ''): v for k, v in state_dict.items() if k.startswith('backbone_i.')}
    # neck_i_new = {k.replace('neck_i.', ''): v for k, v in state_dict.items() if k.startswith('neck_i.')}
    # inf_compressor_new = {k.replace('inf_compressor.', ''): v for k, v in state_dict.items() if k.startswith('inf_compressor.')}
    # # ms_block_inf_new = {k.replace('ms_block_inf.', ''): v for k, v in state_dict.items() if k.startswith('ms_block_inf.')}
    # # dcn_up_conv_i_new = {k.replace('dcn_up_conv_i.', ''): v for k, v in state_dict.items() if k.startswith('dcn_up_conv_i.')}

    # model.backbone_i.load_state_dict(backbone_i_new)
    # model.neck_i.load_state_dict(neck_i_new)
    # model.inf_compressor.load_state_dict(inf_compressor_new)
    # # model.ms_block_inf.load_state_dict(ms_block_inf_new)
    # # model.dcn_up_conv_i.load_state_dict(dcn_up_conv_i_new)

    # neck_i_new_2 = {k.replace('neck_i.', ''): v for k, v in neck_i_new.items()}


    # vqvae_encoder_model_dict = model.vqvae.encoder.state_dict()
    # vqvae_encoder_model_dict_new = {k.replace('vqvae.encoder.', ''): v for k, v in state_dict.items() if k.startswith('vqvae.encoder.')}
    # vqvae_encoder_model_dict.update(vqvae_encoder_model_dict_new)
    # model.vqvae.encoder.load_state_dict(vqvae_encoder_model_dict)

    # vqvae_pre_quantization_conv_model_dict = model.vqvae.pre_quantization_conv.state_dict()
    # vqvae_pre_quantization_conv_model_dict_new = {k.replace('vqvae.pre_quantization_conv.', ''): v for k, v in state_dict.items() if k.startswith('vqvae.pre_quantization_conv.')}
    # vqvae_pre_quantization_conv_model_dict.update(vqvae_pre_quantization_conv_model_dict_new)
    # model.vqvae.pre_quantization_conv.load_state_dict(vqvae_pre_quantization_conv_model_dict)

    # vector_quantization_model_dict = model.vqvae.vector_quantization.state_dict()
    # vector_quantization_model_dict_new = {k.replace('vqvae.vector_quantization.', ''): v for k, v in state_dict.items() if k.startswith('vqvae.vector_quantization.')}
    # vector_quantization_model_dict.update(vector_quantization_model_dict_new)
    # model.vqvae.vector_quantization.load_state_dict(vector_quantization_model_dict)


    # decoder_model_dict = model.vqvae.decoder.state_dict()
    # decoder_model_dict_new = {k.replace('vqvae.decoder.', ''): v for k, v in state_dict.items() if k.startswith('vqvae.decoder.')}
    # decoder_model_dict.update(decoder_model_dict_new)
    # model.vqvae.decoder.load_state_dict(decoder_model_dict)


    return model


# def update_backbone_vqvae(model, filename, map_locaiton='cpu'):

#     # from IPython import embed
#     # embed(header='update')
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     filename = '/home/wangz/wangzhe21/VIMI/work_dirs/0515_VIMI_VQVAE_960x540_12e_bs2x1/best_car_3d_0.5_epoch_10.pth'
#     checkpoint = torch.load(filename, map_location = device)
#     state_dict = checkpoint['state_dict']


#     vector_quantization_model_dict = model.vqvae.vector_quantization.state_dict()
#     vector_quantization_model_dict_new = {k.replace('vqvae.vector_quantization.', ''): v for k, v in state_dict.items() if k.startswith('vqvae.vector_quantization.')}
#     vector_quantization_model_dict.update(vector_quantization_model_dict_new)
#     model.vqvae.vector_quantization.load_state_dict(vector_quantization_model_dict)


#     decoder_model_dict = model.vqvae.decoder.state_dict()
#     decoder_model_dict_new = {k.replace('vqvae.decoder.', ''): v for k, v in state_dict.items() if k.startswith('vqvae.decoder.')}
#     decoder_model_dict.update(decoder_model_dict_new)
#     model.vqvae.decoder.load_state_dict(decoder_model_dict)

#     return model

# model1 = Res_block1()
# for p in model1.parameters():
#     print(type(p), p.shape)
# print('conv1子模块:')
# for p in model1.conv1.parameters():
#     print(type(p), p.shape)

# model1 = Res_block1()
# for name, p in model1.named_parameters():
#     print(name, ' :', p.shape)


