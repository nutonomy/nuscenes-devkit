# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Licensed under the Creative Commons [see licence.txt]

import argparse
from typing import Dict, List

from nuscenes import NuScenes

train = \
    ['scene-0001', 'scene-0002', 'scene-0004', 'scene-0005', 'scene-0006', 'scene-0007', 'scene-0008', 'scene-0009',
     'scene-0010', 'scene-0011', 'scene-0019', 'scene-0020', 'scene-0021', 'scene-0022', 'scene-0023', 'scene-0024',
     'scene-0025', 'scene-0026', 'scene-0027', 'scene-0028', 'scene-0029', 'scene-0030', 'scene-0031', 'scene-0032',
     'scene-0033', 'scene-0034', 'scene-0041', 'scene-0042', 'scene-0043', 'scene-0044', 'scene-0045', 'scene-0046',
     'scene-0047', 'scene-0048', 'scene-0049', 'scene-0050', 'scene-0051', 'scene-0052', 'scene-0053', 'scene-0054',
     'scene-0055', 'scene-0056', 'scene-0057', 'scene-0058', 'scene-0059', 'scene-0060', 'scene-0061', 'scene-0062',
     'scene-0063', 'scene-0064', 'scene-0065', 'scene-0066', 'scene-0067', 'scene-0068', 'scene-0069', 'scene-0070',
     'scene-0071', 'scene-0072', 'scene-0073', 'scene-0074', 'scene-0075', 'scene-0076', 'scene-0120', 'scene-0121',
     'scene-0122', 'scene-0123', 'scene-0124', 'scene-0125', 'scene-0126', 'scene-0127', 'scene-0128', 'scene-0129',
     'scene-0130', 'scene-0131', 'scene-0132', 'scene-0133', 'scene-0134', 'scene-0135', 'scene-0138', 'scene-0139',
     'scene-0149', 'scene-0150', 'scene-0151', 'scene-0152', 'scene-0154', 'scene-0155', 'scene-0157', 'scene-0158',
     'scene-0159', 'scene-0160', 'scene-0161', 'scene-0162', 'scene-0163', 'scene-0164', 'scene-0165', 'scene-0166',
     'scene-0167', 'scene-0168', 'scene-0170', 'scene-0171', 'scene-0172', 'scene-0173', 'scene-0174', 'scene-0175',
     'scene-0176', 'scene-0177', 'scene-0178', 'scene-0179', 'scene-0180', 'scene-0181', 'scene-0182', 'scene-0183',
     'scene-0184', 'scene-0185', 'scene-0187', 'scene-0188', 'scene-0190', 'scene-0191', 'scene-0192', 'scene-0193',
     'scene-0194', 'scene-0195', 'scene-0196', 'scene-0199', 'scene-0200', 'scene-0202', 'scene-0203', 'scene-0204',
     'scene-0206', 'scene-0207', 'scene-0208', 'scene-0209', 'scene-0210', 'scene-0211', 'scene-0212', 'scene-0213',
     'scene-0214', 'scene-0218', 'scene-0219', 'scene-0220', 'scene-0222', 'scene-0224', 'scene-0225', 'scene-0226',
     'scene-0227', 'scene-0228', 'scene-0229', 'scene-0230', 'scene-0231', 'scene-0232', 'scene-0233', 'scene-0234',
     'scene-0235', 'scene-0236', 'scene-0237', 'scene-0238', 'scene-0239', 'scene-0240', 'scene-0241', 'scene-0242',
     'scene-0243', 'scene-0244', 'scene-0245', 'scene-0246', 'scene-0247', 'scene-0248', 'scene-0249', 'scene-0250',
     'scene-0251', 'scene-0252', 'scene-0253', 'scene-0254', 'scene-0255', 'scene-0256', 'scene-0257', 'scene-0258',
     'scene-0259', 'scene-0260', 'scene-0261', 'scene-0262', 'scene-0263', 'scene-0264', 'scene-0283', 'scene-0284',
     'scene-0285', 'scene-0286', 'scene-0287', 'scene-0288', 'scene-0289', 'scene-0290', 'scene-0291', 'scene-0292',
     'scene-0293', 'scene-0294', 'scene-0295', 'scene-0296', 'scene-0297', 'scene-0298', 'scene-0299', 'scene-0300',
     'scene-0301', 'scene-0302', 'scene-0303', 'scene-0304', 'scene-0305', 'scene-0306', 'scene-0315', 'scene-0316',
     'scene-0317', 'scene-0318', 'scene-0321', 'scene-0323', 'scene-0324', 'scene-0328', 'scene-0347', 'scene-0348',
     'scene-0349', 'scene-0350', 'scene-0351', 'scene-0352', 'scene-0353', 'scene-0354', 'scene-0355', 'scene-0356',
     'scene-0357', 'scene-0358', 'scene-0359', 'scene-0360', 'scene-0361', 'scene-0362', 'scene-0363', 'scene-0364',
     'scene-0365', 'scene-0366', 'scene-0367', 'scene-0368', 'scene-0369', 'scene-0370', 'scene-0371', 'scene-0372',
     'scene-0373', 'scene-0374', 'scene-0375', 'scene-0376', 'scene-0377', 'scene-0378', 'scene-0379', 'scene-0380',
     'scene-0381', 'scene-0382', 'scene-0383', 'scene-0384', 'scene-0385', 'scene-0386', 'scene-0388', 'scene-0389',
     'scene-0390', 'scene-0391', 'scene-0392', 'scene-0393', 'scene-0394', 'scene-0395', 'scene-0396', 'scene-0397',
     'scene-0398', 'scene-0399', 'scene-0400', 'scene-0401', 'scene-0402', 'scene-0403', 'scene-0405', 'scene-0406',
     'scene-0407', 'scene-0408', 'scene-0410', 'scene-0411', 'scene-0412', 'scene-0413', 'scene-0414', 'scene-0415',
     'scene-0416', 'scene-0417', 'scene-0418', 'scene-0419', 'scene-0420', 'scene-0421', 'scene-0422', 'scene-0423',
     'scene-0424', 'scene-0425', 'scene-0426', 'scene-0427', 'scene-0428', 'scene-0429', 'scene-0430', 'scene-0431',
     'scene-0432', 'scene-0433', 'scene-0434', 'scene-0435', 'scene-0436', 'scene-0437', 'scene-0438', 'scene-0439',
     'scene-0440', 'scene-0441', 'scene-0442', 'scene-0443', 'scene-0444', 'scene-0445', 'scene-0446', 'scene-0447',
     'scene-0448', 'scene-0449', 'scene-0450', 'scene-0451', 'scene-0452', 'scene-0453', 'scene-0454', 'scene-0455',
     'scene-0456', 'scene-0457', 'scene-0458', 'scene-0459', 'scene-0461', 'scene-0462', 'scene-0463', 'scene-0464',
     'scene-0465', 'scene-0467', 'scene-0468', 'scene-0469', 'scene-0471', 'scene-0472', 'scene-0474', 'scene-0475',
     'scene-0476', 'scene-0477', 'scene-0478', 'scene-0479', 'scene-0480', 'scene-0499', 'scene-0500', 'scene-0501',
     'scene-0502', 'scene-0504', 'scene-0505', 'scene-0506', 'scene-0507', 'scene-0508', 'scene-0509', 'scene-0510',
     'scene-0511', 'scene-0512', 'scene-0513', 'scene-0514', 'scene-0515', 'scene-0517', 'scene-0518', 'scene-0525',
     'scene-0526', 'scene-0527', 'scene-0528', 'scene-0529', 'scene-0530', 'scene-0531', 'scene-0532', 'scene-0533',
     'scene-0534', 'scene-0535', 'scene-0536', 'scene-0537', 'scene-0538', 'scene-0539', 'scene-0541', 'scene-0542',
     'scene-0543', 'scene-0544', 'scene-0545', 'scene-0546', 'scene-0566', 'scene-0568', 'scene-0570', 'scene-0571',
     'scene-0572', 'scene-0573', 'scene-0574', 'scene-0575', 'scene-0576', 'scene-0577', 'scene-0578', 'scene-0580',
     'scene-0582', 'scene-0583', 'scene-0584', 'scene-0585', 'scene-0586', 'scene-0587', 'scene-0588', 'scene-0589',
     'scene-0590', 'scene-0591', 'scene-0592', 'scene-0593', 'scene-0594', 'scene-0595', 'scene-0596', 'scene-0597',
     'scene-0598', 'scene-0599', 'scene-0600', 'scene-0639', 'scene-0640', 'scene-0641', 'scene-0642', 'scene-0643',
     'scene-0644', 'scene-0645', 'scene-0646', 'scene-0647', 'scene-0648', 'scene-0649', 'scene-0650', 'scene-0651',
     'scene-0652', 'scene-0653', 'scene-0654', 'scene-0655', 'scene-0656', 'scene-0657', 'scene-0658', 'scene-0659',
     'scene-0660', 'scene-0661', 'scene-0662', 'scene-0663', 'scene-0664', 'scene-0665', 'scene-0666', 'scene-0667',
     'scene-0668', 'scene-0669', 'scene-0670', 'scene-0671', 'scene-0672', 'scene-0673', 'scene-0674', 'scene-0675',
     'scene-0676', 'scene-0677', 'scene-0678', 'scene-0679', 'scene-0681', 'scene-0683', 'scene-0684', 'scene-0685',
     'scene-0686', 'scene-0687', 'scene-0688', 'scene-0689', 'scene-0695', 'scene-0696', 'scene-0697', 'scene-0698',
     'scene-0700', 'scene-0701', 'scene-0703', 'scene-0704', 'scene-0705', 'scene-0706', 'scene-0707', 'scene-0708',
     'scene-0709', 'scene-0710', 'scene-0711', 'scene-0712', 'scene-0713', 'scene-0714', 'scene-0715', 'scene-0716',
     'scene-0717', 'scene-0718', 'scene-0719', 'scene-0726', 'scene-0727', 'scene-0728', 'scene-0730', 'scene-0731',
     'scene-0733', 'scene-0734', 'scene-0735', 'scene-0736', 'scene-0737', 'scene-0738', 'scene-0739', 'scene-0740',
     'scene-0741', 'scene-0744', 'scene-0746', 'scene-0747', 'scene-0749', 'scene-0750', 'scene-0751', 'scene-0752',
     'scene-0757', 'scene-0758', 'scene-0759', 'scene-0760', 'scene-0761', 'scene-0762', 'scene-0763', 'scene-0764',
     'scene-0765', 'scene-0767', 'scene-0768', 'scene-0769', 'scene-0786', 'scene-0787', 'scene-0789', 'scene-0790',
     'scene-0791', 'scene-0792', 'scene-0803', 'scene-0804', 'scene-0805', 'scene-0806', 'scene-0808', 'scene-0809',
     'scene-0810', 'scene-0811', 'scene-0812', 'scene-0813', 'scene-0815', 'scene-0816', 'scene-0817', 'scene-0819',
     'scene-0820', 'scene-0821', 'scene-0822', 'scene-0847', 'scene-0848', 'scene-0849', 'scene-0850', 'scene-0851',
     'scene-0852', 'scene-0853', 'scene-0854', 'scene-0855', 'scene-0856', 'scene-0858', 'scene-0860', 'scene-0861',
     'scene-0862', 'scene-0863', 'scene-0864', 'scene-0865', 'scene-0866', 'scene-0868', 'scene-0869', 'scene-0870',
     'scene-0871', 'scene-0872', 'scene-0873', 'scene-0875', 'scene-0876', 'scene-0877', 'scene-0878', 'scene-0880',
     'scene-0882', 'scene-0883', 'scene-0884', 'scene-0885', 'scene-0886', 'scene-0887', 'scene-0888', 'scene-0889',
     'scene-0890', 'scene-0891', 'scene-0892', 'scene-0893', 'scene-0894', 'scene-0895', 'scene-0896', 'scene-0897',
     'scene-0898', 'scene-0899', 'scene-0900', 'scene-0901', 'scene-0902', 'scene-0903', 'scene-0945', 'scene-0947',
     'scene-0949', 'scene-0952', 'scene-0953', 'scene-0955', 'scene-0956', 'scene-0957', 'scene-0958', 'scene-0959',
     'scene-0960', 'scene-0961', 'scene-0975', 'scene-0976', 'scene-0977', 'scene-0978', 'scene-0979', 'scene-0980',
     'scene-0981', 'scene-0982', 'scene-0983', 'scene-0984', 'scene-0988', 'scene-0989', 'scene-0990', 'scene-0991',
     'scene-0992', 'scene-0994', 'scene-0995', 'scene-0996', 'scene-0997', 'scene-0998', 'scene-0999', 'scene-1000',
     'scene-1001', 'scene-1002', 'scene-1003', 'scene-1004', 'scene-1005', 'scene-1006', 'scene-1007', 'scene-1008',
     'scene-1009', 'scene-1010', 'scene-1011', 'scene-1012', 'scene-1013', 'scene-1014', 'scene-1015', 'scene-1016',
     'scene-1017', 'scene-1018', 'scene-1019', 'scene-1020', 'scene-1021', 'scene-1022', 'scene-1023', 'scene-1024',
     'scene-1025', 'scene-1044', 'scene-1045', 'scene-1046', 'scene-1047', 'scene-1048', 'scene-1049', 'scene-1050',
     'scene-1051', 'scene-1052', 'scene-1053', 'scene-1054', 'scene-1055', 'scene-1056', 'scene-1057', 'scene-1058',
     'scene-1074', 'scene-1075', 'scene-1076', 'scene-1077', 'scene-1078', 'scene-1079', 'scene-1080', 'scene-1081',
     'scene-1082', 'scene-1083', 'scene-1084', 'scene-1085', 'scene-1086', 'scene-1087', 'scene-1088', 'scene-1089',
     'scene-1090', 'scene-1091', 'scene-1092', 'scene-1093', 'scene-1094', 'scene-1095', 'scene-1096', 'scene-1097',
     'scene-1098', 'scene-1099', 'scene-1100', 'scene-1101', 'scene-1102', 'scene-1104', 'scene-1105', 'scene-1106',
     'scene-1107', 'scene-1108', 'scene-1109', 'scene-1110']

val = \
    ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']

test = \
    ['scene-0077', 'scene-0078', 'scene-0079', 'scene-0080', 'scene-0081', 'scene-0082', 'scene-0083', 'scene-0084',
     'scene-0085', 'scene-0086', 'scene-0087', 'scene-0088', 'scene-0089', 'scene-0090', 'scene-0091', 'scene-0111',
     'scene-0112', 'scene-0113', 'scene-0114', 'scene-0115', 'scene-0116', 'scene-0117', 'scene-0118', 'scene-0119',
     'scene-0140', 'scene-0142', 'scene-0143', 'scene-0144', 'scene-0145', 'scene-0146', 'scene-0147', 'scene-0148',
     'scene-0265', 'scene-0266', 'scene-0279', 'scene-0280', 'scene-0281', 'scene-0282', 'scene-0307', 'scene-0308',
     'scene-0309', 'scene-0310', 'scene-0311', 'scene-0312', 'scene-0313', 'scene-0314', 'scene-0333', 'scene-0334',
     'scene-0335', 'scene-0336', 'scene-0337', 'scene-0338', 'scene-0339', 'scene-0340', 'scene-0341', 'scene-0342',
     'scene-0343', 'scene-0481', 'scene-0482', 'scene-0483', 'scene-0484', 'scene-0485', 'scene-0486', 'scene-0487',
     'scene-0488', 'scene-0489', 'scene-0490', 'scene-0491', 'scene-0492', 'scene-0493', 'scene-0494', 'scene-0495',
     'scene-0496', 'scene-0497', 'scene-0498', 'scene-0547', 'scene-0548', 'scene-0549', 'scene-0550', 'scene-0551',
     'scene-0601', 'scene-0602', 'scene-0603', 'scene-0604', 'scene-0606', 'scene-0607', 'scene-0608', 'scene-0609',
     'scene-0610', 'scene-0611', 'scene-0612', 'scene-0613', 'scene-0614', 'scene-0615', 'scene-0616', 'scene-0617',
     'scene-0618', 'scene-0619', 'scene-0620', 'scene-0621', 'scene-0622', 'scene-0623', 'scene-0624', 'scene-0827',
     'scene-0828', 'scene-0829', 'scene-0830', 'scene-0831', 'scene-0833', 'scene-0834', 'scene-0835', 'scene-0836',
     'scene-0837', 'scene-0838', 'scene-0839', 'scene-0840', 'scene-0841', 'scene-0842', 'scene-0844', 'scene-0845',
     'scene-0846', 'scene-0932', 'scene-0933', 'scene-0935', 'scene-0936', 'scene-0937', 'scene-0938', 'scene-0939',
     'scene-0940', 'scene-0941', 'scene-0942', 'scene-0943', 'scene-1026', 'scene-1027', 'scene-1028', 'scene-1029',
     'scene-1030', 'scene-1031', 'scene-1032', 'scene-1033', 'scene-1034', 'scene-1035', 'scene-1036', 'scene-1037',
     'scene-1038', 'scene-1039', 'scene-1040', 'scene-1041', 'scene-1042', 'scene-1043']

mini_train = \
    ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']

mini_val = \
    ['scene-0103', 'scene-0916']


def create_splits_logs(split: str, nusc: 'NuScenes') -> List[str]:
    """
    Returns the logs in each dataset split of nuScenes.
    Note: Previously this script included the teaser dataset splits. Since new scenes from those logs were added and
          others removed in the full dataset, that code is incompatible and was removed.
    :param split: NuScenes split.
    :param nusc: NuScenes instance.
    :return: A mapping from split name to a list of logs in that split.
    """

    # Load splits on a scene-level.
    scene_splits = create_splits_scenes(verbose=False)

    assert split in scene_splits.keys(), 'Requested split {} which is not a known nuScenes split.'.format(split)

    # Check compatibility of split with nusc versions
    version = nusc.version
    if split in {'train', 'val'}:
        assert version.endswith('trainval'), 'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), 'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split == 'test':
        assert version.endswith('test'), 'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    else:
        raise ValueError('Requested split {} which this function cannot map to logs.'.format(split))

    # Get logs for this split
    scene_to_log = {scene['name']: nusc.get('log', scene['log_token'])['logfile'] for scene in nusc.scene}
    logs = set()
    scenes = scene_splits[split]
    for scene in scenes:
        logs.add(scene_to_log[scene])

    return list(logs)


def create_splits_scenes(verbose: bool = False) -> Dict[str, List[str]]:
    """
    Similar to create_splits_logs, but returns a mapping to scene names, rather than log names.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of scenes names in that split.
    """
    # Use hard-coded splits.
    all_scenes = list(set(train + val + test))
    assert len(all_scenes) == 1000, 'Error: Scenes incomplete!'
    scene_splits = {'train': train, 'val': val, 'test': test,
                    'mini_train': mini_train, 'mini_val': mini_val}

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits


if __name__ == '__main__':
    # Settings.
    parser = argparse.ArgumentParser(description='Prints out the scenes for each split.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()
    dataroot = args.dataroot
    version = args.version
    verbose_ = bool(args.verbose)

    # Init.
    nusc = NuScenes(version=version, verbose=verbose_, dataroot=dataroot)

    # Print the scene-level stats.
    scene_splits = create_splits_scenes(verbose=True)
