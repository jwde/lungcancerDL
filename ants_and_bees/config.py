
# Configurations for test runs

CONFIGS = {
    'ants_bees_MIL' : {
        'info' : 'AlexNetConv5 -> MIL -> Sigmoid',
        'model' : 'alexMIL',
        'params' : 'mil_scores',
        'dset_config' : None,
        'dataset' : 'ants_and_bees',

        'epochs' : 10,
        'init_lr' : 0.001,
        'decay' : 0.60,
        'criterion' : {
            'type' : 'bce_sparse'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : True,
    },

    'ants_bees_MIL_113_F' : {
        'info' : 'AlexNetConv5 -> MIL -> Sigmoid',
        'model' : 'alexMIL',
        'params' : 'mil_scores',
        'dset_config' : {
            'size' : 113,
            'peturb_xy': False,
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 50,
        'init_lr' : 0.02,
        'decay' : 0.9,
        'criterion' : {
            'type' : 'bce_sparse'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : True,
    },

    'ants_bees_MIL_113_T' : {
        'info' : 'AlexNetConv5 -> MIL -> Sigmoid',
        'model' : 'alexMIL',
        'params' : 'mil_scores',
        'dset_config' : {
            'size' : 113,
            'peturb_xy': True
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 50,
        'init_lr' : 0.02,
        'decay' : 0.9,
        'criterion' : {
            'type' : 'bce_sparse'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : True,
    },

    'ants_bees_MIL_50_F' : {
        'info' : 'AlexNetConv5 -> MIL -> Sigmoid',
        'model' : 'alexMIL',
        'params' : 'mil_scores',
        'dset_config' : {
            'size' : 50,
            'peturb_xy': False
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 50,
        'init_lr' : 0.002,
        'decay' : .9,
        'criterion' : {
            'type' : 'bce_sparse'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : True,
    },

    'ants_bees_MIL_50_T' : {
        'info' : 'AlexNetConv5 -> MIL -> Sigmoid',
        'model' : 'alexMIL',
        'params' : 'mil_scores',
        'dset_config' : {
            'size' : 50,
            'peturb_xy': True,
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 50,
        'init_lr' : 0.002,
        'decay' : 0.90,
        'criterion' : {
            'type' : 'bce_sparse'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : True,
    },


    'ants_bees_MIL_25_F' : {
        'info' : 'AlexNetConv5 -> MIL -> Sigmoid',
        'model' : 'alexMIL',
        'params' : 'mil_scores',
        'dset_config' : {
            'size' : 25,
            'peturb_xy': False,
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 50,
        'init_lr' : 0.002,
        'decay' : 0.90,
        'criterion' : {
            'type' : 'bce_sparse'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : True,
    },
    'ants_bees_MIL_25_T' : {
        'info' : 'AlexNetConv5 -> MIL -> Sigmoid',
        'model' : 'alexMIL',
        'params' : 'mil_scores',
        'dset_config' : {
            'size' : 25,
            'peturb_xy': True,
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 50,
        'init_lr' : 0.01,
        'decay' : 0.90,
        'criterion' : {
            'type' : 'bce_sparse'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : True,
    },
    'ants_bees_MIL_5_T' : {
        'info' : 'AlexNetConv5 -> MIL -> Sigmoid',
        'model' : 'alexMIL',
        'params' : 'mil_scores',
        'dset_config' : {
            'size' : 5,
            'peturb_xy': True,
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 500,
        'init_lr' : 0.001,
        'decay' : 0.93,
        'criterion' : {
            'type' : 'bce_sparse'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : True,
    },
    'ants_bees_MIL_1_T' : {
        'info' : 'AlexNetConv5 -> MIL -> Sigmoid',
        'model' : 'alexMIL',
        'params' : 'mil_scores',
        'dset_config' : {
            'size' : 1,
            'peturb_xy': True,
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 10,
        'init_lr' : 0.001,
        'decay' : 0.60,
        'criterion' : {
            'type' : 'bce_sparse'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : True,
    },
    'ants_bees_FC' : {
        'info' : 'AlexNetConv5 -> FC -> Sigmoid',
        'model' : 'alex',
        'params' : 'predict',
        'dset_config' : None,
        'dataset' : 'ants_and_bees',

        'epochs' : 10,
        'init_lr' : 0.001,
        'decay' : 0.60,
        'criterion' : {
            'type' : 'bce'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : False,
    },

    'ants_bees_FC_113_F' : {
        'info' : 'AlexNetConv5 -> FC -> Sigmoid',
        'model' : 'alex',
        'params' : 'predict',
        'dset_config' : {
            'size' : 113,
            'peturb_xy': False,
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 10,
        'init_lr' : 0.001,
        'decay' : 0.60,
        'criterion' : {
            'type' : 'bce'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : False,
    },

    'ants_bees_FC_113_T' : {
        'info' : 'AlexNetConv5 -> FC -> Sigmoid',
        'model' : 'alex',
        'params' : 'predict',
        'dset_config' : {
            'size' : 113,
            'peturb_xy': True
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 10,
        'init_lr' : 0.001,
        'decay' : 0.60,
        'criterion' : {
            'type' : 'bce'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : False,
    },

    'ants_bees_FC_50_F' : {
        'info' : 'AlexNetConv5 -> FC -> Sigmoid',
        'model' : 'alex',
        'params' : 'predict',
        'dset_config' : {
            'size' : 50,
            'peturb_xy': False
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 10,
        'init_lr' : 0.001,
        'decay' : 0.60,
        'criterion' : {
            'type' : 'bce'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : False,
    },

    'ants_bees_FC_50_T' : {
        'info' : 'AlexNetConv5 -> FC -> Sigmoid',
        'model' : 'alex',
        'params' : 'predict',
        'dset_config' : {
            'size' : 50,
            'peturb_xy': True,
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 10,
        'init_lr' : 0.001,
        'decay' : 0.60,
        'criterion' : {
            'type' : 'bce'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : False,
    },


    'ants_bees_FC_25_F' : {
        'info' : 'AlexNetConv5 -> FC -> Sigmoid',
        'model' : 'alex',
        'params' : 'predict',
        'dset_config' : {
            'size' : 25,
            'peturb_xy': False,
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 10,
        'init_lr' : 0.001,
        'decay' : 0.60,
        'criterion' : {
            'type' : 'bce'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : False,
    },
    'ants_bees_FC_25_T' : {
        'info' : 'AlexNetConv5 -> FC -> Sigmoid',
        'model' : 'alex',
        'params' : 'predict',
        'dset_config' : {
            'size' : 25,
            'peturb_xy': True,
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 50,
        'init_lr' : 0.001,
        'decay' : 0.60,
        'criterion' : {
            'type' : 'bce'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : False,
    },
    'ants_bees_FC_5_T' : {
        'info' : 'AlexNetConv5 -> FC -> Sigmoid',
        'model' : 'alex',
        'params' : 'predict',
        'dset_config' : {
            'size' : 5,
            'peturb_xy': True,
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 500,
        'init_lr' : 0.001,
        'decay' : 0.93,
        'criterion' : {
            'type' : 'bce'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : False,
    },
    'ants_bees_FC_1_T' : {
        'info' : 'AlexNetConv5 -> FC -> Sigmoid',
        'model' : 'alex',
        'params' : 'predict',
        'dset_config' : {
            'size' : 1,
            'peturb_xy': True,
        },
        'dataset' : 'ants_and_bees',

        'epochs' : 10,
        'init_lr' : 0.001,
        'decay' : 0.60,
        'criterion' : {
            'type' : 'bce'
        },
        'weight_reg' : 0,
        'batch_size' : 4,
        'is_tuple' : False,
    }



}

#CONFIGS = {
#    'simple' : {
#        'net': models.Simple,
#        'crop' : ((30,33), (0,227), (0,227)),
#    },
#    '3d': {
#        'net' : models.Cnn3d,
#        'crop' : ((0,60),(0,224),(0,224)),
#        'batch_size' : 3,
#    },
#    'alex3d' :{
#        'net' : models.Alex3d,
#        'crop' : ((0,60),(0,227),(0,227)),
#        'params': 'predict',
#        'batch_size' : 1, #ONLY WORKS WITH BATCHSIZE 1
#
#    },
#    'vgg3d' : {
#        'net' : vgg3d.get_pretrained_2D_layers,
#        'batch_size' : 3,
#    },
#    'alexslicer' :{
#        'net' : slicewise_models.Alex,
#        'params': 'predict',
#        'lr_scheduler' : util.exp_lr_decay(0.0001, 0.75),
#        'init_lr' : 0.0001
#        'decay'
#        'batch_size' : 4,
#        'augment_data': False
#    },
#    'alexslicerMIL':{
#        'net' : slicewise_models.AlexMIL,
#        'params': 'mil_scores',
#        'lr': 0.004, # overfits 20 examples with LR 0.004
#        #'lr_scheduler' : util.exp_lr_decay(0.00001, 0.85),
#        'batch_size' : 20,
#        'augment_data': False,
#        'criterion' : lambda y, t: util.sparse_BCE_loss(y, t, 0.00001),
#        'get_probs' : lambda outs: outs[0],
#    },
#    'resnet50' : {
#        'net' : lambda: slicewise_models.ResNet(50),
#        'crop' : ((0,60),(0,225),(0,225)),
#        'params': 'predict',
#        'lr': 0.000005,
#        'batch_size': 4,
#        'lr_scheduler': util.exp_lr_decay(0.000005, 0.95),
#        'augment_data': True
#    },
#    'resnet50MIL' : {
#        'net' : lambda: slicewise_models.ResNetMIL(50),
#        'crop' : ((0,60),(0,224),(0,224)),
#        'params': 'mil_scores',
#        'lr': 0.0005,
#        'batch_size': 4,
#        'lr_scheduler': util.exp_lr_decay(0.0005, 0.95),
#        'augment_data': False,
#        'criterion' : lambda y, t: util.sparse_BCE_loss(y, t, 0.00001),
#        'get_probs' : lambda outs: outs[0],
#
#    },
#    'resnet152' : {
#        'net' : lambda: slicewise_models.ResNet(152),
#        'crop' : ((0,60),(0,225),(0,225)),
#        'params': 'predict',
#        'lr': 0.000005,
#        'lr_scheduler': util.exp_lr_decay(0.000005, 0.95),
#        'batch_size': 4,
#        'augment_data': True,
#        'reg': 0.0001
#    },
#    'resnet152boosted' : {
#        'net': lambda: slicewise_models.ResNetBoosted(152),
#        'crop' : ((0,60),(0,225),(0,225)),
#        'xgboost': True,
#        'batch_size': 4,
#        'augment_data': False,
#        'max_depth': 2
#    },
#    'alexboosted': {
#        'net': lambda: slicewise_models.AlexBoosted(),
#        'xgboost': True,
#        'batch_size': 4,
#        'augment_data': False,
#        'max_depth': 3
#    }
#}
#
class Config(object):
    def __init__(self, net):
        return CONFIG[net]

