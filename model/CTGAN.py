import torch.nn as nn
import torch
import math
from model.model_component import *
from model.FE import Feature_Extractor, S1_Feature_Extractor

class Conformer_Module(nn.Module):
    def __init__(self, dim):
        super(Conformer_Module,self).__init__()
        self.comformer = ConformerBlock(
        dim = dim ** 2,
        dim_head = 1024,
        heads = 16,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.1,
        ff_dropout = 0.1,
        conv_dropout = 0.1
        )

    def forward(self, emb):
        # out: (batch size, length, d_model)
        # out = self.prenet(emb)
        # out: (length, batch size, d_model)
        out = emb.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.comformer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        dim = int(math.sqrt(out.shape[2]))

        return out.view(out.shape[0], out.shape[1], dim, dim)

class CTGAN_Generator(nn.Module):
    def __init__(self, image_size):
        ngf = 32
        super(CTGAN_Generator,self).__init__()
        self.feature_extractor = Feature_Extractor()
        self.downsampling = nn.Sequential(
            nn.Conv2d(2*ngf, 4 * ngf, kernel_size=3, stride=2, padding=1, bias=False,),
            nn.BatchNorm2d(4 * ngf),
            nn.ReLU(True),
            nn.Conv2d(4 * ngf, 8 * ngf, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8 * ngf),
            nn.ReLU(True),
        )
        
        self.transformer = Conformer_Module(image_size // 4)
        self.model_final = nn.Sequential(
            nn.ConvTranspose2d(24 * ngf, 12 * ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(12 * ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(12 * ngf, 6 * ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(6 * ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(6 * ngf, 4, kernel_size=7, padding=0),
            nn.Tanh()
        )

        
    def forward(self, input):
        x0, x1, x2 = input[0], input[1], input[2]
        att0, out0, pred_0 = self.feature_extractor(x0)
        att1, out1, pred_1 = self.feature_extractor(x1)
        att2, out2, pred_2 = self.feature_extractor(x2)
        out0_1 = torch.cat((out0, out1), 1)
        out0_2 = torch.cat((out0, out2), 1)
        out1_2 = torch.cat((out1, out2), 1)
        out0_1 = self.downsampling(out0_1)
        out0_2 = self.downsampling(out0_2)
        out1_2 = self.downsampling(out1_2)
        output = torch.cat((out0_1, out0_2, out1_2), 1)
        output = self.transformer(output.view(output.shape[0], output.shape[1], -1))
        out = self.model_final(output)

        return out, [att0, att1, att2], [pred_0, pred_1, pred_2]


class S1CTGAN_Generator(nn.Module):

    def __init__(self, image_size):
        ngf = 32
        super(S1CTGAN_Generator, self).__init__()
        self.feature_extractor = Feature_Extractor()
        self.s1_feature_extractor = S1_Feature_Extractor()
        self.downsampling = nn.Sequential(
            nn.Conv2d(4 * ngf, 4 * ngf, kernel_size=3, stride=2, padding=1, bias=False, ),
            nn.BatchNorm2d(4 * ngf),
            nn.ReLU(True),
            nn.Conv2d(4 * ngf, 8 * ngf, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8 * ngf),
            nn.ReLU(True),
        )
        self.downsampling_now = nn.Sequential(
            nn.Conv2d(1 * ngf, 4 * ngf, kernel_size=3, stride=2, padding=1, bias=False, ),
            nn.BatchNorm2d(4 * ngf),
            nn.ReLU(True),
            nn.Conv2d(4 * ngf, 8 * ngf, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8 * ngf),
            nn.ReLU(True),
        )

        self.transformer = Conformer_Module(image_size // 4)
        self.model_final = nn.Sequential(
            nn.ConvTranspose2d(32 * ngf, 12 * ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(12 * ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(12 * ngf, 6 * ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(6 * ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(6 * ngf, 4, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def load_feature_extractor(self, filepath):

        fe_state_dict = self.feature_extractor.state_dict()
        pretrained_state_dict = torch.load(
            filepath,
            map_location=next(self.feature_extractor.parameters()).device
        )
        pretrained_fe_state_dict = {
            key.replace("feature_extractor.", ""): value
            for key, value
            in pretrained_state_dict.items()
            if key.replace("feature_extractor.", "") in fe_state_dict
        }

        if not pretrained_fe_state_dict.keys() == fe_state_dict.keys():
            raise ValueError("Could not load weights for feature extractor")

        self.feature_extractor.load_state_dict(pretrained_fe_state_dict)

    def forward(self, input_s2, input_s1, s1_now):

        x0, x1, x2 = input_s2
        s1_0, s1_1, s1_2 = input_s1
        s1_now = s1_now

        att0, out0, pred_0 = self.feature_extractor(x0)
        att1, out1, pred_1 = self.feature_extractor(x1)
        att2, out2, pred_2 = self.feature_extractor(x2)

        out0_s1, pred_0_s1 = self.s1_feature_extractor(s1_0)
        out1_s1, pred_1_s1 = self.s1_feature_extractor(s1_1)
        out2_s1, pred_2_s1 = self.s1_feature_extractor(s1_2)

        out_now_s1, pred_now_s1 = self.s1_feature_extractor(s1_now)

        out0_1 = torch.cat((out0, out0_s1, out1, out1_s1), 1)
        out0_2 = torch.cat((out0, out0_s1, out2, out2_s1), 1)
        out1_2 = torch.cat((out1, out1_s1, out2, out2_s1), 1)

        out0_1 = self.downsampling(out0_1)
        out0_2 = self.downsampling(out0_2)
        out1_2 = self.downsampling(out1_2)
        out_now_s1 = self.downsampling_now(out_now_s1)

        output = torch.cat((out0_1, out0_2, out1_2, out_now_s1), 1)
        output = self.transformer(output.view(output.shape[0], output.shape[1], -1))
        out = self.model_final(output)

        return out, [att0, att1, att2], [pred_0, pred_1, pred_2], [pred_0_s1, pred_1_s1, pred_2_s1]


class CTGAN_Discriminator(nn.Module):
    def __init__(self, input_nc= 3*4+4, ndf=64, n_layers=3):
        super(CTGAN_Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, input):
        return self.discriminator(input)
    

if __name__ == '__main__':
    image_size = 224
    model = CTGAN_Generator(image_size).cuda()
    input1 = torch.randn(2, 4, image_size, image_size).cuda()
    input2 = torch.randn(2, 4, image_size, image_size).cuda()
    input3 = torch.randn(2, 4, image_size, image_size).cuda()
    out, att, _ = model([input1, input2, input3])
    print(out.shape)
