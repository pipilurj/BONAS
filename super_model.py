from opendomain_utils.operations import *
from opendomain_utils.genotypes import PRIMITIVES
from opendomain_utils import genotypes
import numpy as np


class MixedOp(nn.Module):
    '''mask: op'''

    def __init__(self, C, stride, mask):
        super(MixedOp, self).__init__()
        self.stride = stride
        self._ops = nn.ModuleList()
        mask_1 = np.nonzero(mask)[0]
        self._super_mask = mask_1
        if len(mask_1) != 0:
            for selected in np.nditer(mask_1):
                primitive = PRIMITIVES[selected]
                op = OPS[primitive](C, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                self._ops.append(op)

    def forward(self, x, mask):
        if (mask==0).all():
            if self.stride == 1:
                # return x.mul(0.)
                return torch.zeros_like(x)
            # return x[:,:,::self.stride,::self.stride].mul(0.)
            return torch.zeros_like(x[:,:,::self.stride,::self.stride])
        else:
            result = 0
            mask_2 = np.nonzero(mask)[0]
            if len(mask_2) != 0:
                for selected in np.nditer(mask_2):
                    pos = np.where(self._super_mask==selected)[0][0]
                    result += self._ops[pos](x)
            return result


class Cell(nn.Module):
    '''mask: 14 * 8'''

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, mask):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        cnt = 0
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, mask[cnt])
                self._ops.append(op)
                cnt += 1

    def forward(self, s0, s1, mask):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j].forward(h, mask[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3, mask=None):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        # self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, mask)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, mask):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell.forward(s0, s1, mask)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


def geno2mask(genotype):
    des = -1
    mask = np.zeros([14, 8])
    op_names, indices = zip(*genotype.normal)
    for cnt, (name, index) in enumerate(zip(op_names, indices)):
        if cnt % 2 == 0:
            des += 1
            total_state = sum(i + 2 for i in range(des))
        op_idx = PRIMITIVES.index(name)
        node_idx = index + total_state
        mask[node_idx, op_idx] = 1
    return mask


def merge(subnet_masks):
    supernet_mask = np.zeros((14, 8))
    for mask in subnet_masks:
        supernet_mask = mask + supernet_mask
    return supernet_mask


if __name__ == "__main__":
    num_ops = 8
    steps = 4
    arch = "BONAS"
    genotype = eval("genotypes.%s" % arch)
    mask = geno2mask(genotype)

    print(str(mask))
    print(hash(mask.tostring()))
    k = sum(1 for i in range(steps) for n in range(2 + i))
    mask = torch.ones(k, num_ops)

    '''---------------------------------------------'''
    mask = np.array([[0, 1, 0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1]]
                    )
    # print(mask)
    mix_op = MixedOp(C=16, stride=1, mask=mask[0])
    cell = Cell(steps, multiplier=4, C_prev_prev=16, C_prev=16, C=16, reduction=False, reduction_prev=False, mask=mask)
