from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(
            params) + f'\nKeras-style summary:\n{self.summary()}'

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def default_input_size(self):
        try:
            return self._DEFAULT_INPUT_SIZE
        except AttributeError:
            raise NotImplementedError('Define _DEFAULT_INPUT_SIZE or override default_input_size')

    def summary(self, input_size=None, batch_size=-1, device="cuda"):
        """Based on https://github.com/sksq96/pytorch-summary"""

        input_size = input_size or self.default_input_size

        def register_hook(module):

            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == self)
            ):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 2 for batchnorm
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        self.apply(register_hook)

        # make a forward pass
        self(*x)

        # remove these hooks
        for h in hooks:
            h.remove()

        summary_lines = []
        summary_lines.append("----------------------------------------------------------------")
        summary_lines.append("{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #"))
        summary_lines.append("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            summary_lines.append(line_new)

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        summary_lines.append("================================================================")
        summary_lines.append("Total params: {0:,}".format(total_params))
        summary_lines.append("Trainable params: {0:,}".format(trainable_params))
        summary_lines.append("Non-trainable params: {0:,}".format(total_params - trainable_params))
        summary_lines.append("----------------------------------------------------------------")
        summary_lines.append("Input size (MB): %0.2f" % total_input_size)
        summary_lines.append("Forward/backward pass size (MB): %0.2f" % total_output_size)
        summary_lines.append("Params size (MB): %0.2f" % total_params_size)
        summary_lines.append("Estimated Total Size (MB): %0.2f" % total_size)
        summary_lines.append("----------------------------------------------------------------")

        return '\n'.join(summary_lines)
