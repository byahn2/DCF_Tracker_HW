import torch.nn as nn
import torch


class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)


class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        # wf: the fourier transformation of correlation kernel w. You will need to calculate the best wf in update method.
        self.wf = None
        # xf: the fourier transformation of target patch x.
        self.xf = None
        self.config = config

    def forward(self, z):
        """
        :param z: the multiscale searching patch. Shape (num_scale, 3, crop_sz, crop_sz)
        :return response: the response of cross correlation. Shape (num_scale, 1, crop_sz, crop_sz)

        You are required to calculate response using self.wf to do cross correlation on the searching patch z
        """
        # obtain feature of z and add hanning window
        z = self.feature(z) * self.config.cos_window

        num_scale, channels, crop_sz, crop_sz = z.shape
        zf = torch.rfft(z,2)
        w_star = self.wf.clone().detach()
        w_star[:,:,:,:,1] = w_star[:,:,:,:,1]*-1
        output = torch.cuda.FloatTensor(num_scale, 1, crop_sz, crop_sz//2+1, 2).fill_(0)
        for c in range(num_scale):
            for l in range(channels):
                temp = torch.mul(w_star[0, 1, :, :, :], zf[c, l, :, :, :])
                out_real, out_imag = self.imag_mult(w_star[0, 1, :, :, 0], w_star[0, 1, :, :, 1], zf[c, l, :, :, 0],
                                                    zf[c, l, :, :, 1])
                temp[:, :, 0] = out_real
                temp[:, :, 1] = out_imag
                output[c, 0, :, :, :] += temp
        response = torch.irfft(output, 2)
        return response

    def imag_mult(self, matrix_a_real, matrix_a_imag, matrix_b_real, matrix_b_imag):
        out_real = matrix_a_real * matrix_b_real - matrix_a_imag * matrix_b_imag
        out_imag = matrix_a_real * matrix_b_imag + matrix_a_imag * matrix_b_real
        return out_real, out_imag

    def imag_div(self, matrix_a_real, matrix_a_imag, matrix_b_real, matrix_b_imag):
        out_real = (matrix_a_real * matrix_b_real - matrix_a_imag * matrix_b_imag) / (
                    torch.mul(matrix_b_real, matrix_b_real) + torch.mul(matrix_b_imag, matrix_b_imag))
        out_imag = (matrix_a_real * matrix_b_imag + matrix_a_imag * matrix_b_real) / (
                    torch.mul(matrix_b_real, matrix_b_real) + torch.mul(matrix_b_imag, matrix_b_imag))
        return out_real, out_imag



    def update(self, x, lr=1.0):
        """
        this is the to get the fourier transformation of  optimal correlation kernel w
        :param x: the input target patch (1, 3, h ,w)
        :param lr: the learning rate to update self.xf and self.wf

        The other arguments concealed in self.config that will be used here:
        -- self.config.cos_window: the hanning window applied to the x feature. Shape (crop_sz, crop_sz),
                                   where crop_sz is 125 in default.
        -- self.config.yf: the fourier transform of idea gaussian response. Shape (1, 1, crop_sz, crop_sz//2+1, 2)
        -- self.config.lambda0: the coefficient of the normalize term.

        things you need to calculate:
        -- self.xf: the fourier transformation of x. Shape (1, channel, crop_sz, crop_sz//2+1, 2)
        -- self.wf: the fourier transformation of optimal correlation filter w, calculated by the formula,
                    Shape (1, channel, crop_sz, crop_sz//2+1, 2)
        """
        # x: feature of patch x with hanning window. Shape (1, 32, crop_sz, crop_sz)
        # torch.fft(input, signal_ndim, normalized=False)
        x = self.feature(x) * self.config.cos_window
        scale_size, channels, crop_sz, crop_sz = x.shape
        xf = torch.rfft(x,2)
        if type(self.xf) == type(None):
            self.xf = xf
        else:
            self.xf = (1-lr)*self.xf.data + lr*xf
        y_star = self.config.yf.clone().detach()
        y_star[:, :, :, :, 1] = y_star[:, :, :, :, 1] * -1
        numerator = torch.mul(self.xf, y_star)
        out_real, out_imag = self.imag_mult(self.xf[:, :, :, :, 0], self.xf[:, :, :, :, 1],
                                            self.config.yf[:, :, :, :, 0], self.config.yf[:, :, :, :, 1])
        numerator[:, :, :, :, 0] = out_real
        numerator[:, :, :, :, 1] = out_imag
        denominator = torch.cuda.FloatTensor(1, channels, crop_sz, crop_sz // 2 + 1, 2).fill_(0)
        phi_K = torch.cuda.FloatTensor(1, crop_sz, crop_sz // 2 + 1, 2).fill_(0)
        for k in range(channels):
            phi_K[:, :, :, :] = self.xf[:, k, :, :, :]
            conj_phi_K = phi_K.clone().detach()
            conj_phi_K[:, :, :, 1] = conj_phi_K[:, :, :, 1] * -1
            final_prod = phi_K * conj_phi_K
            out_real, out_imag = self.imag_mult(phi_K[:, :, :, 0], phi_K[:, :, :, 1], conj_phi_K[:, :, :, 0],
                                                conj_phi_K[:, :, :, 1])
            final_prod[:, :, :, 0] = out_real
            final_prod[:, :, :, 1] = out_imag
            denominator[:, 0, :, :, :] += final_prod + self.config.lambda0
        for k in range(channels):
            denominator[:, k, :, :, :] = denominator[:, 0, :, :, :]
        wf = numerator / denominator
        out_real, out_imag = self.imag_div(numerator[:, :, :, :, 0], numerator[:, :, :, :, 1],
                                           denominator[:, :, :, :, 0], denominator[:, :, :, :, 1])
        wf[:, :, :, :, 0] = out_real
        wf[:, :, :, :, 1] = out_imag
        if type(self.wf) == type(None):
            self.wf = wf
        else:
            self.wf = (1-lr)*self.wf.data + lr*wf






    def load_param(self, path='param.pth'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict']
            if 'module' in state_dict.keys()[0]:  # train with nn.DataParallel
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                self.load_state_dict(new_state_dict)
            else:
                self.load_state_dict(state_dict)
        else:
            self.feature.load_state_dict(checkpoint)

