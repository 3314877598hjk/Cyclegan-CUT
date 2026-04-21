import torch

from .base_model import BaseModel
from . import networks


class CUTModel(BaseModel):
    """Contrastive Unpaired Translation with edge preservation and attention.

    This model upgrades the CycleGAN training objective from bidirectional
    cycle consistency to one-way adversarial learning plus PatchNCE. It keeps
    the existing ResNet generator, optional self-attention module, and adds a
    Sobel edge loss for structure-sensitive tasks such as map/vector transfer.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if not is_train:
            parser.set_defaults(dataset_mode="single")

        parser.add_argument("--CUT_mode", type=str, default="CUT", choices=["CUT", "FastCUT"], help="CUT uses identity NCE; FastCUT is lighter and uses stronger NCE")
        parser.add_argument("--nce_layers", type=str, default="4,8,12,16", help="comma-separated generator layer ids used for PatchNCE")
        parser.add_argument("--num_patches", type=int, default=256, help="number of patches sampled from each feature map for PatchNCE")
        parser.add_argument("--nce_T", type=float, default=0.07, help="temperature for PatchNCE")
        parser.add_argument("--lambda_GAN", type=float, default=1.0, help="weight for adversarial generator loss")
        parser.add_argument("--lambda_NCE", type=float, default=None, help="weight for PatchNCE; defaults to 1 for CUT and 10 for FastCUT")
        parser.add_argument("--nce_idt", action="store_true", help="force identity-domain PatchNCE even in FastCUT mode")
        parser.add_argument("--no_nce_idt", action="store_true", help="disable identity-domain PatchNCE")
        parser.add_argument("--no_attention", action="store_true", help="disable self-attention module in the generator")

        if is_train:
            parser.add_argument("--use_edge_loss", action="store_true", help="keep Sobel edge loss enabled for compatibility with older scripts")
            parser.add_argument("--no_edge_loss", action="store_true", help="disable Sobel edge structure loss")
            parser.add_argument("--lambda_edge", type=float, default=1.0, help="weight for Sobel edge structure loss")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.nce_layers = [int(layer_id) for layer_id in opt.nce_layers.split(",") if layer_id.strip()]
        if opt.lambda_NCE is None:
            opt.lambda_NCE = 10.0 if opt.CUT_mode == "FastCUT" else 1.0
        self.nce_idt = opt.nce_idt or (opt.CUT_mode == "CUT" and not opt.no_nce_idt)
        self.use_edge_loss = self.isTrain and (not opt.no_edge_loss) and opt.lambda_edge > 0

        if self.isTrain:
            self.loss_names = ["G_GAN", "NCE"]
            if self.nce_idt:
                self.loss_names.append("NCE_Y")
            if self.use_edge_loss:
                self.loss_names.append("edge")
            self.loss_names += ["G", "D_real", "D_fake", "D"]
            self.visual_names = ["real_A", "fake_B", "real_B"]
            if self.nce_idt:
                self.visual_names.append("idt_B")
            self.model_names = ["G", "D"]
        else:
            self.loss_names = []
            self.visual_names = ["real_A", "fake_B"]
            self.model_names = ["G"]

        use_attention = not getattr(opt, "no_attention", False)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, use_attention=use_attention)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            self.netF = networks.PatchSampleF(use_l2_norm=True)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = networks.PatchNCELoss(opt.nce_T).to(self.device)
            if self.use_edge_loss:
                self.criterionEdge = networks.EdgeLoss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        if "B" in input:
            AtoB = self.opt.direction == "AtoB"
            self.real_A = input["A" if AtoB else "B"].to(self.device)
            self.real_B = input["B" if AtoB else "A"].to(self.device)
            self.image_paths = input["A_paths" if AtoB else "B_paths"]
        else:
            self.real_A = input["A"].to(self.device)
            self.image_paths = input["A_paths"]

    def forward(self):
        if self.isTrain and self.nce_idt:
            real = torch.cat((self.real_A, self.real_B), dim=0)
            fake = self.netG(real)
            self.fake_B = fake[: self.real_A.size(0)]
            self.idt_B = fake[self.real_A.size(0) :]
        else:
            self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        pred_fake = self.netD(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        self.loss_D.backward()

    def calculate_NCE_loss(self, src, tgt):
        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        batch_size = src.shape[0]
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.criterionNCE(f_q, f_k, batch_size).mean()
            total_nce_loss = total_nce_loss + loss
        return total_nce_loss / len(feat_q_pool)

    def backward_G(self):
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True) * self.opt.lambda_GAN

        self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B) * self.opt.lambda_NCE
        if self.nce_idt:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B) * self.opt.lambda_NCE
            loss_nce = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_nce = self.loss_NCE

        if self.use_edge_loss:
            self.loss_edge = self.criterionEdge(self.fake_B, self.real_A) * self.opt.lambda_edge
        else:
            self.loss_edge = 0.0

        self.loss_G = self.loss_G_GAN + loss_nce + self.loss_edge
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
