import torch

def select_optimal_group(fh: torch.Tensor, masks: torch.Tensor, d):
    """Select the group with the lowest h value

    Args:
        fh (torch.Tensor): flattened h, h - h_{G_optimal}
        last_optimal_group (set): As the name implies
        index (list): Groups

    Returns:
        torch.Tensor : onehot_optimal_group
    """

    Group_h = torch.mm(masks.float(), fh.reshape(d, 1)) # [G,1]
    min_h_group_num = torch.argmin(Group_h)
    optimal_group = masks[min_h_group_num].clone()
    masks[min_h_group_num]=0

    return optimal_group

def greedy_project(h: torch.Tensor, delta: torch.Tensor, masks: torch.Tensor, groups_num: int) -> torch.Tensor:
    """ Select k groups using greedy projection, i.e. select the first k groups with the lowest h value

    Args:
        h (torch.Tensor): the standard value of greedy selecting
        delta (torch.Tensor): the original perturbations
        masks (torch.Tensor): onehot_groups_index
        index: groups_index

    Returns:
        torch.Tensor: the perturbation
    """
    u = delta
    flatten_h = torch.flatten(h) # [3072]
    flatten_u = torch.flatten(u) # [3072]
    flatten_v = torch.zeros_like(flatten_u)
    d = masks.shape[1]

    for i in range(groups_num):
        onehot_optimal_group = select_optimal_group(flatten_h, masks, d)
        flatten_v += onehot_optimal_group * flatten_u
        flatten_h *= onehot_optimal_group.logical_not() 
        flatten_u *= onehot_optimal_group.logical_not() 
    return flatten_v

def standard_grouping(image_size, filtersize, stride, channels, d):
    R = torch.floor((image_size - filtersize) / stride) + 1
    R = R.type(torch.int32)
    C = R
    index = torch.zeros([R*C, filtersize * filtersize * channels],dtype=torch.int32)
    masks = []
    tmpidx = 0
    for r in range(R):
        plus1 = r * stride * image_size * channels
        for c in range(C):
            index_ = []
            for i in range(filtersize):
                plus2 = c * stride * channels + i * image_size * channels + plus1
                index_.append(torch.arange(plus2, plus2 + filtersize * channels))
            
            li = []
            for x in range(d):
                for j in range(filtersize):
                    if x in index_[j]:
                        li.append(1)
                        break
                else:
                    li.append(0)
            masks.append(li)

            # masks.append([ 1 if x in index_[0] or x in index_[1] else 0 for x in range(d) ])
            index[tmpidx] = torch.cat(index_, dim=-1)
            tmpidx += 1
    # index = torch.tile(index, (batch_size,1,1))
    return torch.tensor(masks)
