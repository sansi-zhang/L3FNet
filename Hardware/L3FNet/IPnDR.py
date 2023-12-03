import numpy as np

def cat(tensors, dim=0):
    return np.concatenate(tensors, axis=dim)

def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    # Get the input shape
    batch_size, channels, height, width = input.shape

    # Fill in the input
    input_padded = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # Calculate the height and width of the output
    output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Create a zero array to hold the results
    output = np.zeros((batch_size, channels, kernel_size, kernel_size, output_height, output_width))

    # Iterate over each pixel
    for i in range(kernel_size):
        for j in range(kernel_size):
            start_i = i * dilation
            start_j = j * dilation
            end_i = start_i + stride * output_height
            end_j = start_j + stride * output_width

            output[:, :, i, j, :, :] = input_padded[:, :, start_i:end_i:stride, start_j:end_j:stride]

    return output.reshape(batch_size, channels * kernel_size * kernel_size, output_height, output_width)


def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    # Get the input shape
    batch_size, _, output_height, output_width = input.shape
    channels = input.shape[1] // (kernel_size * kernel_size)

    # Create a zero array to hold the results
    output = np.zeros((batch_size, channels, output_size[0], output_size[1]))

    # Iterate over each pixel
    for i in range(kernel_size):
        for j in range(kernel_size):
            start_i = i * dilation
            start_j = j * dilation
            end_i = start_i + stride * output_height
            end_j = start_j + stride * output_width

            output[:, :, start_i:end_i:stride, start_j:end_j:stride] += input[:, i * kernel_size + j::kernel_size * kernel_size, :, :]

    return output



def SAI2MacPI_plus(x, angRes):
    # x:torch.Size([4, 1, 432, 432])
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes     # h=w=48
    mindisp = -4
    maxdisp = 4
    # Calculate the MacPI for d=0
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(cat(tempV, dim=3))
    input = cat(tempU, dim=2)
    
    # MacPI is computed for all d based on d=0
    temp = []
    for d in range(mindisp, maxdisp + 1):
        if d < 0:
            dilat = int(abs(d) * angRes + 1)
            pad = int(0.5 * angRes * (angRes - 1) * abs(d))
        if d == 0:
            dilat = 1
            pad = 0
        if d > 0:
            dilat = int(abs(d) * angRes - 1)
            pad = int(0.5 * angRes * (angRes - 1) * abs(d) - angRes + 1)
        mid = unfold(input, kernel_size=angRes, dilation=dilat, padding=pad, stride=angRes)
        print(mid.shape)
        out_d = fold(mid, output_size=(hu,wv), kernel_size=angRes, dilation=1, padding=0, stride=angRes)
        print(out_d.shape)
        temp.append(out_d)
    out = cat(temp, dim=1)
    return out


class Regression:
    def __init__(self, mindisp, maxdisp):
        self.maxdisp = maxdisp
        self.mindisp = mindisp

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, cost):
        cost = np.squeeze(cost, axis=1)
        score = self.softmax(cost)              # B, D, H, W
        temp = np.zeros(score.shape)            # B, D, H, W
        for d in range(self.maxdisp - self.mindisp + 1):
            temp[:, d, :, :] = score[:, d, :, :] * (self.mindisp + d)
        disp = np.sum(temp, axis=1, keepdims=True)     # B, 1, H, W
        return disp