import torch
import torch.nn as nn

class ReduLayer_N_X_M(nn.Module):

    def __init__(self, num_classes, n_dim, epsilon = 0.1, lambda_softmax = 0.1, learning_rate = 0.01):
        super(ReduLayer_N_X_M,self).__init__()
        self.num_classes = num_classes
        self.n_dim = n_dim
        self.epsilon = epsilon
        self.lambda_softmax = lambda_softmax
        self.learning_rate = learning_rate
        self.bn_fun = nn.BatchNorm1d(self.num_classes, track_running_stats = False, affine = False)
        #储存C类对应的z@z.T的和
        self.EC_proto = nn.Parameter(torch.zeros(self.num_classes, self.n_dim, self.n_dim).float(), requires_grad=False)
        #储存每一类对应的计数
        self.EC_count = nn.Parameter(torch.zeros(self.num_classes), requires_grad=False)
    
    def forward(self, x, return_pi = False):
        #x:B x N x M
        x = self.bn_fun(x)
        e_x_z = self.get_E_params(x)
        bool_fix_EC_count_depth, c_x_z, c_x_z_norm = self.get_C_params(x)
        #pi_j: B x C
        pi_j = (nn.Softmax(dim=1)(-1*self.lambda_softmax*c_x_z_norm))*bool_fix_EC_count_depth
        # C x 1 x 1
        gama_j = self.get_gama_j().to(x.device)
        # B x N x M
        this_z_d = e_x_z - (gama_j*c_x_z*pi_j.unsqueeze(-1).unsqueeze(-1)).sum(1)
        next_z = x + self.learning_rate*this_z_d
        if return_pi:
            return next_z, pi_j
        else:
            return next_z
    
    def get_e_alpha(self):
        #计算e的alpha
        if self.EC_count.sum() > 1e-6:
            e_alpha = self.n_dim/(self.epsilon*self.epsilon*self.EC_count.sum())
        else:
            e_alpha = torch.Tensor([0.])
        return e_alpha
    
    def get_c_alpha(self):
        #计算c的alpha，对没有数据的项标记
        bool_fix_EC_count_depth = self.EC_count.bool().float()
        gyaku_bool_fix_EC_count_depth = 1 - bool_fix_EC_count_depth
        EC_count_zero_plus_one = self.EC_count + gyaku_bool_fix_EC_count_depth
        c_alpha_pre_fix = self.n_dim/(EC_count_zero_plus_one*self.epsilon*self.epsilon)
        c_alpha = c_alpha_pre_fix * bool_fix_EC_count_depth
        c_alpha = c_alpha.reshape(self.num_classes, 1, 1)
        return c_alpha, bool_fix_EC_count_depth

    def get_gama_j(self):
        #计算gama
        if self.EC_count.sum() > 1e-6:
            gama_j = self.EC_count / self.EC_count.sum()
        else:
            gama_j = torch.zeros(self.EC_count.shape)
        return gama_j.unsqueeze(-1).unsqueeze(-1)

    def get_E_or_C(self, z_z_T, alpha):
        #计算E和C根据z@z.T的不同决定返回的矩阵
        eye = torch.eye(self.n_dim).to(z_z_T.device)
        return alpha * torch.inverse(alpha*z_z_T + eye)

    def get_c_x_z_norm(self, c_x_z):
        c_x_z_norm = torch.norm(c_x_z.reshape(c_x_z.shape[0], c_x_z.shape[1], c_x_z.shape[2]* c_x_z.shape[3]), p=2, dim=2)
        c_x_z_norm = 10*c_x_z_norm/c_x_z_norm.sum(1).unsqueeze(-1)
        return c_x_z_norm

    def get_E_params(self, x):
        #计算e的项
        e_alpha = self.get_e_alpha().to(x.device)
        #e:N x N
        e = self.get_E_or_C(self.EC_proto.sum(0), e_alpha)
        #计算Ez_i: B x N x M
        e_x_z = e@x
        return e_x_z
    
    def get_C_params(self, x):
        #计算c的项
        c_alpha, bool_fix_EC_count_depth = self.get_c_alpha()
        #c:C x N x N
        c = self.get_E_or_C(self.EC_proto, c_alpha)
        #计算Cz_i: B x C x N x M
        c_x_z = (c.unsqueeze(0))@(x.unsqueeze(1))
        #预测
        c_x_z_norm = self.get_c_x_z_norm(c_x_z)
        return bool_fix_EC_count_depth, c_x_z, c_x_z_norm

if __name__ == "__main__":
    model = ReduLayer_N_X_M(2,3)
    x = torch.ones(2,3,5)
    print(model(x))