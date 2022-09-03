import yaml

# data = {'set_param': {'img_path': 'Data/val2017',
#                       'anno_path': 'Data/annotations_trainval2017/annotations/instances_val2017.json'},
#         'loader_param': {'shuffle': False,
#                          'batch_size': 16}}
data = {'backbone_param': {'name': 'resnet50',
                           'train_backbone': False,
                           'return_interm_layers': False},
        'detr_param': {'num_cls': 91,
                       'num_layers': 6,
                       'embed_size': 256,
                       'heads': 8,
                       'dropout': 0,
                       'forward_expansion': 4},
        'device': 'cuda',
        'loss': {'lambda_cls': 1,
                 'lambda_iou': 2,
                 'lambda_L1': 5,
                 'topK': 5}}
with open('model_config.yml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)
