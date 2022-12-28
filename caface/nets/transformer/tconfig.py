default = {
    'depths': [3],
    'num_group_tokens': [4],
    'intermediate_outdim': 'same',
    'key_query_combined_info_propagate': True,
    'confidence_dim': 1,
    'random_group_discard': 0.0,
    'num_heads': [8],
    'norm_dim': 64
}

small = {
    'depths': [2],
    'num_group_tokens': [4],
    'intermediate_outdim': 'same',
    'key_query_combined_info_propagate': True,
    'confidence_dim': 1,
    'random_group_discard': 0.0,
    'num_heads': [8],
    'norm_dim': 64
}



def return_config(name):
    assert 'cat' in name
    config = default
    if 'small' in name:
        config = small
    if 'g1' in name:
        config['num_group_tokens'] = [1]
    if 'g2' in name:
        config['num_group_tokens'] = [2]
    if 'g4' in name:
        config['num_group_tokens'] = [4]
    if 'g8' in name:
        config['num_group_tokens'] = [8]
    if 'g16' in name:
        config['num_group_tokens'] = [16]
    if 'g32' in name:
        config['num_group_tokens'] = [32]
    if 'g64' in name:
        config['num_group_tokens'] = [64]

    if 'conf512' in name:
        config['confidence_dim'] = 512

    print('Cluster Aggregate Config')
    print(config)

    return config
