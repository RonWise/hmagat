from typing import Sequence
from collections import OrderedDict
import math

import torch
import torch.nn.functional as F

from hmagat.generate_additional_data import any_additional_data
from hmagat.modules.model.model_selection import get_gnn_module

HYPEREDGE_LOSS_ID = -1


class CNN(torch.nn.Module):
    convs: Sequence[torch.nn.Conv2d]
    batch_norms: Sequence[torch.nn.BatchNorm2d]
    compressMLP: Sequence[torch.nn.Linear]

    def __init__(
        self,
        *,
        numChannel,
        numStride,
        convW,
        convH,
        nMaxPoolFilterTaps,
        numMaxPoolStride,
        embedding_sizes,
    ):
        super().__init__()

        convs = []
        batch_norms = []
        numConv = len(numChannel) - 1
        nFilterTaps = [3] * numConv
        nPaddingSzie = [1] * numConv
        for l in range(numConv):
            convs.append(
                torch.nn.Conv2d(
                    in_channels=numChannel[l],
                    out_channels=numChannel[l + 1],
                    kernel_size=nFilterTaps[l],
                    stride=numStride[l],
                    padding=nPaddingSzie[l],
                    bias=True,
                )
            )
            batch_norms.append(torch.nn.BatchNorm2d(num_features=numChannel[l + 1]))
            # convl.append(torch.nn.ReLU(inplace=True))

            convW = (
                int((convW - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l]) + 1
            )
            convH = (
                int((convH - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l]) + 1
            )
            # Adding maxpooling
            if l % 2 == 0:
                convW = int((convW - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                convH = int((convH - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                # http://cs231n.github.io/convolutional-networks/

        self.convs = torch.nn.ModuleList(convs)
        self.batch_norms = torch.nn.ModuleList(batch_norms)

        numFeatureMap = numChannel[-1] * convW * convH

        numCompressFeatures = [numFeatureMap] + embedding_sizes

        self.nMaxPoolFilterTaps = nMaxPoolFilterTaps
        self.numMaxPoolStride = numMaxPoolStride

        compressmlp = []
        for l in range(len(embedding_sizes)):
            compressmlp.append(
                torch.nn.Linear(
                    in_features=numCompressFeatures[l],
                    out_features=numCompressFeatures[l + 1],
                    bias=True,
                )
            )
        self.compressMLP = torch.nn.ModuleList(compressmlp)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()
        for lin in self.compressMLP:
            lin.reset_parameters()

    def forward(self, x):
        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x)
            x = batch_norm(x)
            x = F.relu(x)

            if i % 2 == 0:
                x = F.max_pool2d(
                    x, kernel_size=self.nMaxPoolFilterTaps, stride=self.numMaxPoolStride
                )
        x = x.reshape((x.shape[0], -1))
        for lin in self.compressMLP:
            x = lin(x)
            x = F.relu(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        dilation=(1, 1),
        residual=True,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(
            inplanes, planes, stride, padding=dilation[0], dilation=dilation[0]
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=dilation[1])
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        self.conv2.reset_parameters()
        self.bn2.reset_parameters()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):
    def __init__(
        self,
        layers,
        in_channels=3,
        num_classes=128,
        channels=(32, 64, 128, 128),
        out_map=False,
        out_middle=False,
        pool_size=2,
        arch="D",
    ):
        super(ResNet, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        self.conv1 = torch.nn.Conv2d(
            in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(channels[0])
        self.relu = torch.nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], stride=2)
        self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=1)
        self.layer3 = self._make_layer(BasicBlock, channels[2], layers[2], stride=1)

        if num_classes > 0:
            self.avgpool = torch.nn.AvgPool2d(pool_size)
            self.fc = torch.nn.Conv2d(
                self.out_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True
            )
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self, block, planes, blocks, stride=1, dilation=1, new_level=True, residual=True
    ):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                dilation=(
                    (1, 1)
                    if dilation == 1
                    else (dilation // 2 if new_level else dilation, dilation)
                ),
                residual=residual,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    residual=residual,
                    dilation=(dilation, dilation),
                )
            )

        return torch.nn.Sequential(*layers)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        for l in self.layer1:
            l.reset_parameters()
        for l in self.layer2:
            l.reset_parameters()
        for l in self.layer3:
            l.reset_parameters()
        self.fc.reset_parameters()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        y = list()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        y.append(x)

        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        # print("x - postCNN:", x.shape)
        if self.out_map:
            x = self.fc(x)
        else:
            x_TMP = self.avgpool(x)
            # print('x - after avgpool', x_TMP.shape)
            x = self.fc(x_TMP)
            # print('x - after FCN', x.shape)
            # print('check x == x_tmp', x-x_TMP)
            # x = x.view(x.size(0), -1)

        if self.out_middle:
            return x, y
        else:
            return x


class ResNetLarge_withMLP(torch.nn.Module):
    def __init__(self, output_size, embedding_sizes, in_channels=3):
        super().__init__()
        self.resnet = ResNet([1, 1, 1], in_channels=in_channels, out_map=False)
        self.dropout = torch.nn.Dropout(0.2)
        self.lin = torch.nn.Linear(
            in_features=1152, out_features=output_size, bias=True
        )

        numFeatureMap = output_size
        numCompressFeatures = [numFeatureMap] + embedding_sizes

        compressmlp = []
        for l in range(len(embedding_sizes)):
            compressmlp.append(
                torch.nn.Linear(
                    in_features=numCompressFeatures[l],
                    out_features=numCompressFeatures[l + 1],
                    bias=True,
                )
            )
        self.compressMLP = torch.nn.ModuleList(compressmlp)

    def reset_parameters(self):
        self.resnet.reset_parameters()
        self.lin.reset_parameters()
        for lin in self.compressMLP:
            lin.reset_parameters()

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = x.reshape((x.shape[0], -1))
        x = self.lin(x)

        for lin in self.compressMLP:
            x = lin(x)
            x = F.relu(x)
        return x


class ResNetSmaller_withMLP(torch.nn.Module):
    def __init__(self, output_size, embedding_sizes, in_channels=3):
        super().__init__()
        self.resnet = ResNet(
            [1, 1, 1],
            in_channels=in_channels,
            out_map=False,
            channels=(16, 32, 64, 64),
            num_classes=64,
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.lin = torch.nn.Linear(in_features=576, out_features=output_size, bias=True)

        numFeatureMap = output_size
        numCompressFeatures = [numFeatureMap] + embedding_sizes

        compressmlp = []
        for l in range(len(embedding_sizes)):
            compressmlp.append(
                torch.nn.Linear(
                    in_features=numCompressFeatures[l],
                    out_features=numCompressFeatures[l + 1],
                    bias=True,
                )
            )
        self.compressMLP = torch.nn.ModuleList(compressmlp)

    def reset_parameters(self):
        self.resnet.reset_parameters()
        self.lin.reset_parameters()
        for lin in self.compressMLP:
            lin.reset_parameters()

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = x.reshape((x.shape[0], -1))
        x = self.lin(x)

        for lin in self.compressMLP:
            x = lin(x)
            x = F.relu(x)
        return x


class ResNetVSmaller_withMLP(torch.nn.Module):
    def __init__(self, output_size, embedding_sizes, in_channels=3):
        super().__init__()
        self.resnet = ResNet(
            [1, 1, 1],
            in_channels=in_channels,
            out_map=False,
            channels=(8, 16, 32, 32),
            num_classes=32,
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.lin = torch.nn.Linear(in_features=288, out_features=output_size, bias=True)

        numFeatureMap = output_size
        numCompressFeatures = [numFeatureMap] + embedding_sizes

        compressmlp = []
        for l in range(len(embedding_sizes)):
            compressmlp.append(
                torch.nn.Linear(
                    in_features=numCompressFeatures[l],
                    out_features=numCompressFeatures[l + 1],
                    bias=True,
                )
            )
        self.compressMLP = torch.nn.ModuleList(compressmlp)

    def reset_parameters(self):
        self.resnet.reset_parameters()
        self.lin.reset_parameters()
        for lin in self.compressMLP:
            lin.reset_parameters()

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = x.reshape((x.shape[0], -1))
        x = self.lin(x)

        for lin in self.compressMLP:
            x = lin(x)
            x = F.relu(x)
        return x


def get_gnn_input_pre_processor(input_size):
    aux_data = []
    output_size = input_size

    if len(aux_data) == 0:

        def _processor(x, data):
            return x

        return _processor, input_size

    def _processor(x, data):
        return torch.concatenate([x] + [data[key] for key in aux_data], dim=-1)

    return _processor, output_size


class GNNInputProcessor(torch.nn.Module):
    def __init__(
        self,
        input_size,
        pre_gnn_embedding_size,
        pre_gnn_num_mlp_layers,
    ):
        super().__init__()
        self.preprocessor, intmd_sz = get_gnn_input_pre_processor(input_size=input_size)

        if (pre_gnn_num_mlp_layers is None) or (pre_gnn_num_mlp_layers < 1):
            self.processor = None
            self.output_sz = intmd_sz
        else:
            if pre_gnn_embedding_size is None:
                # No value given, taking intmd_sz
                pre_gnn_embedding_size = intmd_sz
            mlp = [torch.nn.Linear(intmd_sz, pre_gnn_embedding_size)] + [
                torch.nn.Linear(pre_gnn_embedding_size, pre_gnn_embedding_size)
                for _ in range(pre_gnn_num_mlp_layers - 1)
            ]
            self.processor = torch.nn.ModuleList(mlp)
            self.output_sz = pre_gnn_embedding_size

    def get_output_size(self):
        return self.output_sz

    def reset_parameters(self):
        if self.processor is not None:
            for lin in self.processor:
                lin.reset_parameters()

    def forward(self, x, data):
        x = self.preprocessor(x, data)
        if self.processor is not None:
            for lin in self.processor:
                x = lin(x)
                x = torch.nn.functional.relu(x)
        return x


def get_gnn_input_processor(
    input_size,
    pre_gnn_embedding_size,
    pre_gnn_num_mlp_layers,
) -> tuple[GNNInputProcessor, int]:
    module = GNNInputProcessor(
        input_size=input_size,
        pre_gnn_embedding_size=pre_gnn_embedding_size,
        pre_gnn_num_mlp_layers=pre_gnn_num_mlp_layers,
    )
    return module, module.get_output_size()


def get_num_edge_attr_input_channels(use_edge_attr_for_messages: str) -> int:
    if use_edge_attr_for_messages == "positions":
        return 2
    elif use_edge_attr_for_messages == "dist":
        return 1
    elif use_edge_attr_for_messages == "manhattan":
        return 1
    elif use_edge_attr_for_messages == "positions+dist":
        return 3
    elif use_edge_attr_for_messages == "positions+manhattan":
        return 3
    else:
        raise ValueError(f"Unsupported edge_attr method: {use_edge_attr_for_messages}.")


class AdditionalDataProcessor(torch.nn.Module):
    def __init__(
        self,
        additional_data_idx,
        output_size,
        lin_x=False,
    ):
        super().__init__()
        self.greedy_action_mlp = None
        self.prev_actions_mlp = None
        self.lin_x = None
        if additional_data_idx[1] is not None:
            greedy_action_mlp_lins = []
            greedy_action_mlp_sizes = [5, output_size]
            for i in range(len(greedy_action_mlp_sizes) - 1):
                greedy_action_mlp_lins.append(
                    torch.nn.Linear(
                        in_features=greedy_action_mlp_sizes[i],
                        out_features=greedy_action_mlp_sizes[i + 1],
                    )
                )
            self.greedy_action_mlp = torch.nn.ModuleList(greedy_action_mlp_lins)
        if additional_data_idx[2] is not None:
            _, num_prev_actions = additional_data_idx[2]
            prev_actions_mlp_lins = []
            prev_actions_mlp_sizes = [num_prev_actions * 5, output_size, output_size]
            for i in range(len(prev_actions_mlp_sizes) - 1):
                prev_actions_mlp_lins.append(
                    torch.nn.Linear(
                        in_features=prev_actions_mlp_sizes[i],
                        out_features=prev_actions_mlp_sizes[i + 1],
                    )
                )
            self.prev_actions_mlp = torch.nn.ModuleList(prev_actions_mlp_lins)
        if lin_x:
            self.lin_x = torch.nn.Linear(
                in_features=output_size, out_features=output_size
            )

    def reset_parameters(self):
        if self.greedy_action_mlp is not None:
            for lin in self.greedy_action_mlp:
                lin.reset_parameters()
        if self.prev_actions_mlp is not None:
            for lin in self.prev_actions_mlp:
                lin.reset_parameters()
        if self.lin_x is not None:
            self.lin_x.reset_parameters()

    def forward(self, x, data):
        if self.lin_x is not None:
            x = self.lin_x(x)
            x = F.relu(x)
        if self.greedy_action_mlp is not None:
            greedy_action = data.greedy_action
            for lin in self.greedy_action_mlp:
                greedy_action = lin(greedy_action)
                greedy_action = F.relu(greedy_action)
            x = x + greedy_action
        if self.prev_actions_mlp is not None:
            prev_actions = data.prev_actions
            for lin in self.prev_actions_mlp:
                prev_actions = lin(prev_actions)
                prev_actions = F.relu(prev_actions)
            x = x + prev_actions
        return x


class ResetablleSequential(torch.nn.Sequential):
    def reset_parameters(self):
        for module in self:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


def get_cnn(cnn_mode, cnn_num_input_channels, cnn_output_size, inW, inH):
    if cnn_mode == "basic-CNN":
        cnn = CNN(
            numChannel=[cnn_num_input_channels, 32, 32, 64, 64, 128],
            numStride=[1, 1, 1, 1, 1],
            convW=inW,
            convH=inH,
            nMaxPoolFilterTaps=2,
            numMaxPoolStride=2,
            embedding_sizes=[cnn_output_size],
        )
    elif cnn_mode == "ResNetLarge_withMLP":
        cnn = ResNetLarge_withMLP(
            output_size=cnn_output_size,
            embedding_sizes=[cnn_output_size],
            in_channels=cnn_num_input_channels,
        )
    elif cnn_mode == "MLP":
        channels = [cnn_num_input_channels, 32, 32, cnn_output_size]
        mlp = []
        mlp.append(
            torch.nn.Linear(
                in_features=channels[0], out_features=channels[1], bias=True
            )
        )
        for i in range(1, len(channels) - 1):
            mlp.append(torch.nn.ReLU())
            mlp.append(
                torch.nn.Linear(
                    in_features=channels[i],
                    out_features=channels[i + 1],
                    bias=True,
                )
            )
        cnn = ResetablleSequential(*mlp)
    else:
        raise ValueError(f"Unsupported cnn_mode: {cnn_mode}.")

    return cnn


class DecentralPlannerGATNet(torch.nn.Module):
    def __init__(
        self,
        *,
        FOV,
        numInputFeatures,
        num_attention_heads,
        use_dropout,
        gnn_type,
        gnn_kwargs,
        concat_attention,
        num_classes=5,
        cnn_mode="basic-CNN",
        cnn_output_size=None,
        num_gnn_layers=None,
        embedding_sizes_gnn=None,
        use_edge_weights=False,
        use_edge_attr=False,
        edge_dim=None,
        model_residuals=None,
        pre_gnn_embedding_size=None,
        pre_gnn_num_mlp_layers=None,
        module_residual=[],
        additional_data_idx=[None, None, None],
        lin_x_before_additional_data=False,
        use_edge_attr_for_messages=None,
        edge_attr_cnn_mode=None,
        coordination_state_size=0,
        coordination_state_update="gru",
    ):
        super().__init__()

        assert concat_attention is True, "Currently only support concat attention."

        if embedding_sizes_gnn is None:
            assert num_gnn_layers is not None
            embedding_sizes_gnn = num_gnn_layers * [numInputFeatures]
        else:
            if num_gnn_layers is not None:
                assert num_gnn_layers == len(embedding_sizes_gnn)
            else:
                num_gnn_layers = len(embedding_sizes_gnn)

        inW = (FOV + 1) * 2 + 1
        inH = (FOV + 1) * 2 + 1

        if cnn_output_size is None:
            cnn_output_size = numInputFeatures
        cnn_num_input_channels = 3
        if additional_data_idx[0] is not None:
            cnn_num_input_channels = 4

        #####################################################################
        #                                                                   #
        #                CNN to extract feature                             #
        #                                                                   #
        #####################################################################

        self.cnn = get_cnn(
            cnn_mode=cnn_mode,
            cnn_num_input_channels=cnn_num_input_channels,
            cnn_output_size=cnn_output_size,
            inW=inW,
            inH=inH,
        )

        self.use_edge_attr_for_messages = use_edge_attr_for_messages
        self.edge_attr_cnn = None
        if use_edge_attr_for_messages is not None:
            if edge_attr_cnn_mode is None:
                edge_attr_cnn_mode = cnn_mode
            num_edge_attr_input_channels = get_num_edge_attr_input_channels(
                use_edge_attr_for_messages=use_edge_attr_for_messages
            )

            self.edge_attr_cnn = get_cnn(
                cnn_mode=edge_attr_cnn_mode,
                cnn_num_input_channels=num_edge_attr_input_channels,
                cnn_output_size=cnn_output_size,
                inW=inW,
                inH=inH,
            )
            if edge_dim is None:
                edge_dim = cnn_output_size
            assert edge_dim == cnn_output_size

        self.additional_data_processor = AdditionalDataProcessor(
            additional_data_idx=additional_data_idx,
            output_size=cnn_output_size,
            lin_x=lin_x_before_additional_data,
        )

        self.gnn_pre_processor, self.numFeatures2Share = get_gnn_input_processor(
            input_size=cnn_output_size,
            pre_gnn_embedding_size=pre_gnn_embedding_size,
            pre_gnn_num_mlp_layers=pre_gnn_num_mlp_layers,
        )

        #####################################################################
        #                                                                   #
        #                    (hyper)graph neural network                    #
        #                                                                   #
        #####################################################################
        self.hypergraph_generator = None
        self.gnn = get_gnn_module(
            in_channels=self.numFeatures2Share,
            embedding_sizes=embedding_sizes_gnn,
            num_attention_heads=num_attention_heads,
            num_gnn_layers=num_gnn_layers,
            model_type=gnn_type,
            use_edge_weights=use_edge_weights,
            use_edge_attr=use_edge_attr,
            edge_dim=edge_dim,
            model_residuals=model_residuals,
            **gnn_kwargs,
        )

        gnn_output_size = num_attention_heads * embedding_sizes_gnn[-1]
        self.coordination_state_size = coordination_state_size
        self.coordination_state_update = coordination_state_update
        self.coordination_state_cell = None
        self._coordination_state = None
        if self.coordination_state_size > 0:
            if self.coordination_state_update != "gru":
                raise ValueError(
                    "Only GRU coordination-state updates are currently supported."
                )
            self.coordination_state_cell = torch.nn.GRUCell(
                input_size=gnn_output_size,
                hidden_size=self.coordination_state_size,
            )

        #####################################################################
        #                                                                   #
        #                    MLP --- map to actions                         #
        #                                                                   #
        #####################################################################

        actions_input_size = gnn_output_size
        if self.coordination_state_size > 0:
            actions_input_size += self.coordination_state_size

        actions_mlp_sizes = [
            actions_input_size,
            embedding_sizes_gnn[-1],
            num_classes,
        ]

        actionsfc = []
        for i in range(len(actions_mlp_sizes) - 1):
            actionsfc.append(
                torch.nn.Linear(
                    in_features=actions_mlp_sizes[i],
                    out_features=actions_mlp_sizes[i + 1],
                )
            )
        self.use_dropout = use_dropout
        self.actionsMLP = torch.nn.ModuleList(actionsfc)

        self.cnn_to_out_lin = None
        for res in module_residual:
            if res == "cnn-to-out":
                self.cnn_to_out_lin = torch.nn.Linear(
                    cnn_output_size, gnn_output_size
                )
            else:
                raise ValueError(f"Unsupported module_residual: {res}.")
        self.simulation = False

    def reset_parameters(self):
        self.cnn.reset_parameters()
        if self.edge_attr_cnn is not None:
            self.edge_attr_cnn.reset_parameters()
        self.additional_data_processor.reset_parameters()
        self.gnn_pre_processor.reset_parameters()
        if self.hypergraph_generator is not None:
            self.hypergraph_generator.reset_parameters()
        self.gnn.reset_parameters()
        if self.coordination_state_cell is not None:
            self.coordination_state_cell.reset_parameters()
        for lin in self.actionsMLP:
            lin.reset_parameters()
        if self.cnn_to_out_lin is not None:
            self.cnn_to_out_lin.reset_parameters()

    def in_simulation(self, value):
        self.simulation = value
        self.reset_coordination_state()

    def reset_coordination_state(self):
        self._coordination_state = None

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, data, return_pre_gnn_input=False):
        cnn_out = self.cnn(x)
        cnn_out = self.additional_data_processor(cnn_out, data)

        x = self.gnn_pre_processor(cnn_out, data)
        pre_gnn_input = x
        gnn_input_kwargs = dict()

        edge_attr = None
        if self.hypergraph_generator is not None:
            edge_index, edge_attr, edge_weights, _ = self.hypergraph_generator(x, data)
            gnn_input_kwargs["edge_index"] = edge_index
            gnn_input_kwargs["edge_weight"] = edge_weights

        if self.edge_attr_cnn is not None:
            if edge_attr is not None and edge_attr.shape[0] > 0:
                edge_attr = self.edge_attr_cnn(edge_attr)
            elif data.edge_attr is not None and data.edge_attr.shape[0] > 0:
                edge_attr = self.edge_attr_cnn(data.edge_attr)
            else:
                edge_attr = None
            gnn_input_kwargs["edge_attr"] = edge_attr
        x = self.gnn(x, data, **gnn_input_kwargs)

        if self.cnn_to_out_lin is not None:
            res_out = self.cnn_to_out_lin(cnn_out)
            res_out = F.relu(res_out)
            x = res_out + x
        if self.coordination_state_cell is not None:
            x = self._apply_coordination_state(x)
        for lin in self.actionsMLP[:-1]:
            x = lin(x)
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, p=0.2, training=self.training)
        x = self.actionsMLP[-1](x)

        if return_pre_gnn_input:
            return x, pre_gnn_input

        return x

    def _apply_coordination_state(self, x):
        if self.simulation:
            prev_state = self._coordination_state
            if (
                prev_state is None
                or prev_state.shape[0] != x.shape[0]
                or prev_state.device != x.device
                or prev_state.dtype != x.dtype
            ):
                prev_state = x.new_zeros((x.shape[0], self.coordination_state_size))
            state = self.coordination_state_cell(x, prev_state)
            self._coordination_state = state.detach()
        else:
            prev_state = x.new_zeros((x.shape[0], self.coordination_state_size))
            state = self.coordination_state_cell(x, prev_state)
        return torch.cat([x, state], dim=-1)


def _decode_residual_args(args):
    if args.module_residual is None:
        return []

    return args.module_residual.split("+")


_GNN_DEF_KEYS = [
    "imitation_learning_model",
    "embedding_size",
    "num_gnn_layers",
    "num_attention_heads",
    "edge_dim",
    "model_residuals",
    "use_edge_weights",
    "use_edge_attr",
    "hyperedge_feature_generator",
    "use_edge_attr_for_messages",
    "edge_attr_cnn_mode",
    "final_feature_generator",
    "coordination_state_size",
    "coordination_state_update",
]


def _decode_args(args: dict, prefix: str = "") -> dict:
    hypergraph_model = False
    args = {key: args[f"{prefix}{key}"] for key in _GNN_DEF_KEYS}
    model_kwargs = {
        key: args[key]
        for key in [
            "num_gnn_layers",
            "use_edge_attr",
            "use_edge_weights",
            "edge_dim",
            "model_residuals",
            "use_edge_attr_for_messages",
            "edge_attr_cnn_mode",
            "coordination_state_size",
            "coordination_state_update",
        ]
    }
    gnn_kwargs = dict()

    if args["imitation_learning_model"] == "MAGAT":
        gnn_kwargs = gnn_kwargs | {
            "use_edge_attr_for_messages": args["use_edge_attr_for_messages"]
        }
        gnn_type = "MAGAT"
    elif args["imitation_learning_model"] == "DirectionalHMAGAT":
        hypergraph_model = True
        gnn_kwargs = gnn_kwargs | {
            "hyperedge_feature_generator": args["hyperedge_feature_generator"],
            "final_feature_generator": args["final_feature_generator"],
            "use_edge_attr_for_messages": args["use_edge_attr_for_messages"],
        }
        gnn_type = "DirectionalHMAGAT"
    else:
        raise ValueError(
            f"Unsupported imitation learning model {args['imitation_learning_model']}."
        )
    model_kwargs = model_kwargs | {
        "gnn_kwargs": gnn_kwargs,
        "gnn_type": gnn_type,
    }
    return model_kwargs, hypergraph_model


def get_model(args, device) -> tuple[torch.nn.Module, bool, dict]:
    _, additional_data_idx = any_additional_data(args)

    common_kwargs = dict(
        FOV=args.obs_radius,
        numInputFeatures=args.embedding_size,
        num_attention_heads=args.num_attention_heads,
        cnn_mode=args.cnn_mode,
        use_dropout=True,
        concat_attention=True,
        num_classes=5,
        pre_gnn_embedding_size=args.pre_gnn_embedding_size,
        pre_gnn_num_mlp_layers=args.pre_gnn_num_mlp_layers,
        module_residual=_decode_residual_args(args),
        additional_data_idx=additional_data_idx,
        lin_x_before_additional_data=args.lin_x_before_additional_data,
    )
    common_dataset_kwargs = dict()
    dict_args = vars(args)

    model_kwargs, hypergraph_model = _decode_args(dict_args)
    dataset_kwargs = {
        "use_edge_attr": model_kwargs["use_edge_attr"],
        "use_edge_attr_for_messages": args.use_edge_attr_for_messages,
    }

    model = DecentralPlannerGATNet(
        **common_kwargs,
        **model_kwargs,
    ).to(device)
    model.reset_parameters()

    return model, hypergraph_model, dataset_kwargs | common_dataset_kwargs


def load_partial_state_dict(model, state_dict, print_prefix=""):
    model_state = model.state_dict()
    loadable_state = OrderedDict()
    skipped_missing = []
    skipped_shape = []
    skipped_related = []
    shape_mismatch_modules = set()

    for key, value in state_dict.items():
        if key not in model_state:
            skipped_missing.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            shape_mismatch_modules.add(key.rsplit(".", 1)[0])
            continue

    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            continue
        module_name = key.rsplit(".", 1)[0]
        if module_name in shape_mismatch_modules:
            skipped_related.append(key)
            continue
        loadable_state[key] = value

    missing_after_load, unexpected_after_load = model.load_state_dict(
        loadable_state, strict=False
    )

    print(f"{print_prefix}Partial checkpoint loading:")
    print(f"{print_prefix}  loaded keys: {len(loadable_state)}")
    print(f"{print_prefix}  skipped missing keys: {len(skipped_missing)}")
    print(f"{print_prefix}  skipped shape-mismatch keys: {len(skipped_shape)}")
    print(f"{print_prefix}  skipped related keys: {len(skipped_related)}")
    print(f"{print_prefix}  model keys left missing: {len(missing_after_load)}")
    print(f"{print_prefix}  unexpected keys after load: {len(unexpected_after_load)}")

    if skipped_missing:
        print(f"{print_prefix}  skipped missing key names:")
        for key in skipped_missing:
            print(f"{print_prefix}    {key}")

    if skipped_shape:
        print(f"{print_prefix}  skipped shape-mismatch key names:")
        for key, checkpoint_shape, model_shape in skipped_shape:
            print(
                f"{print_prefix}    {key}: checkpoint {checkpoint_shape} -> model {model_shape}"
            )

    if skipped_related:
        print(f"{print_prefix}  skipped keys from shape-mismatched modules:")
        for key in skipped_related:
            print(f"{print_prefix}    {key}")

    if missing_after_load:
        print(f"{print_prefix}  model keys initialized from current model:")
        for key in missing_after_load:
            print(f"{print_prefix}    {key}")

    return {
        "loaded": list(loadable_state.keys()),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "skipped_related": skipped_related,
        "missing_after_load": missing_after_load,
        "unexpected_after_load": unexpected_after_load,
    }
