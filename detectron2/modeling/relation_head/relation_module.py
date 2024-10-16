import torch
from torch import nn
from detectron2.layers import Linear
import fvcore.nn.weight_init as weight_init
from detectron2.utils.registry import Registry
from torch.nn import functional as F

RELATION_MODULE_REGISTER = Registry("REALTION_MODULE")


class RelationBetweenMulti(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(RelationBetweenMulti, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // unit_nums
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.g = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.g.weight, mean=0, std=0.01)
        nn.init.constant_(self.g.bias, 0)

        # 2019/10/23
        # self.W = nn.Linear(self.inter_channels, self.inter_channels)
        # nn.init.normal_(self.W.weight, mean=0, std=0.01)
        # nn.init.constant_(self.W.bias, 0)

        self.theta = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.theta.weight, mean=0, std=0.01)
        nn.init.constant_(self.theta.bias, 0)

        self.phi = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.phi.weight, mean=0, std=0.01)
        nn.init.constant_(self.phi.bias, 0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )
        nn.init.normal_(self.concat_project[0].weight, mean=0, std=0.01)

    def forward(self, x):
        # print(x.shape)
        # exit()
        g_x = self.g(x)

        theta_x = self.theta(x)
        theta_x = theta_x.permute(1, 0)
        N = theta_x.size(1)
        C = theta_x.size(0)
        theta_x = theta_x.view(C, N, 1)
        theta_x = theta_x.repeat(1, 1, N)

        phi_x = self.phi(x)
        phi_x = phi_x.permute(1, 0)
        phi_x = phi_x.view(C, 1, N)
        phi_x = phi_x.repeat(1, N, 1)

        concat_feature = torch.cat((theta_x, phi_x), dim=0)
        concat_feature = concat_feature.view(1, *concat_feature.size()[:])
        f = self.concat_project(concat_feature)
        f = f.view(N, N)
        f_dic_C = f / N

        z = torch.matmul(f_dic_C, g_x)

        return z


class RelationBetweenPair(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(RelationBetweenPair, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // unit_nums
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.g = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.g.weight, mean=0, std=0.01)
        nn.init.constant_(self.g.bias, 0)

        # 2019/10/23
        # self.W = nn.Linear(self.inter_channels, self.inter_channels)
        # nn.init.normal_(self.W.weight, mean=0, std=0.01)
        # nn.init.constant_(self.W.bias, 0)

        self.theta = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.theta.weight, mean=0, std=0.01)
        nn.init.constant_(self.theta.bias, 0)

        self.phi = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.phi.weight, mean=0, std=0.01)
        nn.init.constant_(self.phi.bias, 0)

        self.concat_project = nn.Sequential(
            nn.Linear(self.inter_channels * 2, 1, bias=False),
            nn.ReLU()
        )
        nn.init.normal_(self.concat_project[0].weight, mean=0, std=0.01)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        g_y = self.g(y)

        theta_x = self.theta(x)
        phi_y = self.phi(y)

        concat_feature = torch.cat((theta_x, phi_y), dim=1)
        f = self.concat_project(concat_feature)

        z = self.gamma * f * g_y

        return z


class RelationOnSpatial(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(RelationOnSpatial, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // unit_nums
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.g = nn.Conv2d(256, self.inter_channels, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.g.weight, mean=0, std=0.01)
        nn.init.constant_(self.g.bias, 0)

        # 2019/10/23
        # self.W = nn.Linear(self.inter_channels, self.inter_channels)
        # nn.init.normal_(self.W.weight, mean=0, std=0.01)
        # nn.init.constant_(self.W.bias, 0)

        self.phi = nn.Conv2d(256, self.inter_channels, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.phi.weight, mean=0, std=0.01)
        nn.init.constant_(self.phi.bias, 0)

        self.theta = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.theta.weight, mean=0, std=0.01)
        nn.init.constant_(self.theta.bias, 0)

        self.concat_project = nn.Conv2d(self.inter_channels*2, 1, 1, 1, 0, bias=False)
        nn.init.normal_(self.concat_project.weight, mean=0, std=0.01)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):  # torch.Size([n, 1024]) , torch.Size([1, 256, 7, 7])

        g_y = self.g(y)
        theta_x = self.theta(x) #n*128
        
        N = theta_x.size(0)    # n
        C = theta_x.size(1)    # 128
        #print(N)
        
        phi_y = self.phi(y)                 #[1, 128, 7, 7]
        phi_y = phi_y.repeat(N, 1, 1, 1)    #[n, 128, 7, 7]
         
        resolution = phi_y.size(2)

        theta_x = theta_x.view(N, C, 1, 1)
        theta_x = theta_x.repeat(1, 1, resolution, resolution) #[n, 128, 7, 7]

        concat_feature = torch.cat((theta_x, phi_y), dim=1)    #[n,256, 7, 7]
        f = self.concat_project(concat_feature)                #[n, 1, 7, 7]
        f = f.view(N, -1) #[n, 49]
        f = F.softmax(f, dim=1) #[n, 49]
        
        g_y = g_y.squeeze()
        g_y = g_y.permute(1, 2, 0)
        g_y = g_y.view(resolution*resolution, -1)    #[49, 128]

        z = self.gamma * torch.mm(f, g_y)
        
        return z


class CompressPersonFeature(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(CompressPersonFeature, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // unit_nums

        self.fc = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, person_probs, person_features):
        person_features = self.fc(person_features)
        result = self.gamma * person_features
        return result

class BeforeAfterFeature(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(BeforeAfterFeature, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // unit_nums
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.g = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.g.weight, mean=0, std=0.01)
        nn.init.constant_(self.g.bias, 0)

        self.theta = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.theta.weight, mean=0, std=0.01)
        nn.init.constant_(self.theta.bias, 0)

        self.phi = Linear(self.in_channels, self.inter_channels)
        nn.init.normal_(self.phi.weight, mean=0, std=0.01)
        nn.init.constant_(self.phi.bias, 0)

        self.concat_project = nn.Sequential(
            nn.Linear(self.inter_channels * 2, 1, bias=False),
            nn.ReLU()
        )
        nn.init.normal_(self.concat_project[0].weight, mean=0, std=0.01)

        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x, x_b,x_a): #[n1*1024] ,[n2*1024],[n3*1024]

        x_ab = torch.cat((x_b, x_a), dim=0)

        g_y = self.g(x_ab)       #[n2+n3, 128]
        g_y=torch.mean(g_y,dim=0).unsqueeze(dim=0)
        g_y=g_y.repeat(g_y.size(0),1)
        
        theta_x = self.theta(x)  #[n1,128]
        phi_y = self.phi(x_ab)   #[n2+n3, 128]
        
        phi_y_p=torch.mean(phi_y,dim=0).unsqueeze(dim=0)
        phi_y_p=phi_y_p.repeat(theta_x.size(0),1)

        concat_feature = torch.cat((theta_x, phi_y_p), dim=1)
        f = self.concat_project(concat_feature) 
        
        z = self.gamma * f * g_y

        return z



class ConcatenationUnit(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(ConcatenationUnit, self).__init__()
        self.multi_relation_unit = RelationBetweenMulti(unit_nums, in_channels)
        self.pair_relation_unit = RelationBetweenPair(unit_nums, in_channels)
        self.spatial_relation = RelationOnSpatial(unit_nums, in_channels)
        self.person_prior = CompressPersonFeature(unit_nums, in_channels)
        self.before_after = BeforeAfterFeature(unit_nums, in_channels)
        self.before_after_local1 = RelationBetweenPair(unit_nums, in_channels)
        self.before_after_local2 = RelationBetweenPair(unit_nums, in_channels)

    def forward(self, x, person_probs, person_features,x_bef,x_aft,x_bef_l,x_aft_l):   #return (n,128)
        origin_feature, local_feature, global_feature = x
        # origin_feature_bef, local_feature_bef, global_feature_bef = x_bef
        # origin_feature_aft, local_feature_aft, global_feature_aft = x_aft
        # origin_feature_bef_loacl, local_feature_bef_local, global_feature_bef_local = x_bef_l
        # origin_feature_aft_local, local_feature_aft_local, global_feature_aft_local = x_aft_l

        origin_feature_bef, _, _= x_bef
        origin_feature_aft, _, _ = x_aft
        origin_feature_bef_loacl, _, _ = x_bef_l
        origin_feature_aft_local, _, _ = x_aft_l

        # print(origin_feature.shape)
        # print(origin_feature_bef.shape)
        # print(origin_feature_aft.shape)
        # print(local_feature.shape)
        # print(global_feature.shape)
        #exit()
        origin_relation = self.multi_relation_unit(origin_feature)
        local_relation = self.pair_relation_unit(origin_feature, local_feature)
        global_relation = self.spatial_relation(origin_feature, global_feature)
        # global_relation_b = self.spatial_relation(origin_feature, global_feature_bef)
        # global_relation_a = self.spatial_relation(origin_feature, global_feature_aft)
        #person_prior = self.person_prior(person_probs, person_features)
        before_after_relation = self.before_after(origin_feature,origin_feature_bef,origin_feature_aft)
        before_local_relation = self.before_after_local1(origin_feature,origin_feature_bef_loacl)
        after_local_relation = self.before_after_local2(origin_feature,origin_feature_aft_local)
        #print('aaaaaaa')
        z = origin_relation +local_relation+global_relation +before_local_relation+after_local_relation+before_after_relation
        return z


class ConcatenationUnit_FirstSpatial(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(ConcatenationUnit_FirstSpatial, self).__init__()
        self.multi_relation_unit = RelationBetweenMulti(unit_nums, in_channels)
        self.pair_relation_unit = RelationBetweenPair(unit_nums, in_channels)
        self.spatial_relation = RelationOnSpatial(unit_nums, in_channels)

    def forward(self, x,x_bef,x_aft,t):   #return (n,128)
        origin_feature, local_feature, global_feature = x
        origin_feature_bef, local_feature_bef, global_feature_bef = x_bef
        origin_feature_aft, local_feature_aft, global_feature_aft = x_aft
        if t=='bef':
            origin_relation_bef = self.multi_relation_unit(origin_feature_bef)
            local_relation_bef = self.pair_relation_unit(origin_feature_bef, local_feature_bef)
            global_relation_bef = self.spatial_relation(origin_feature_bef, global_feature_bef)

            z = origin_relation_bef + local_relation_bef + global_relation_bef
        if t=='cur':
            origin_relation = self.multi_relation_unit(origin_feature)
            local_relation = self.pair_relation_unit(origin_feature, local_feature)
            global_relation = self.spatial_relation(origin_feature, global_feature)

            z = origin_relation + local_relation + global_relation
        if t=='aft':
            origin_relation_aft = self.multi_relation_unit(origin_feature_aft)
            local_relation_aft = self.pair_relation_unit(origin_feature_aft, local_feature_aft)
            global_relation_aft = self.spatial_relation(origin_feature_aft, global_feature_aft)

            z = origin_relation_aft + local_relation_aft + global_relation_aft

        return z
    
class ConcatenationUnit_SecondTemporal(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(ConcatenationUnit_SecondTemporal, self).__init__()

        self.before_after = BeforeAfterFeature(unit_nums, in_channels)
        self.before_after_local = RelationBetweenPair(unit_nums, in_channels)

    def forward(self, x,y_bef,y,y_aft,x_bef_l,x_aft_l):   #return (n,128)
        origin_feature, local_feature, global_feature = x
        origin_feature_bef_loacl, local_feature_bef_local, global_feature_bef_local = x_bef_l
        origin_feature_aft_local, local_feature_aft_local, global_feature_aft_local = x_aft_l

        before_after_relation = self.before_after(y,y_bef,y_aft)
        before_local_relation = self.before_after_local(origin_feature,origin_feature_bef_loacl)
        after_local_relation = self.before_after_local(origin_feature,origin_feature_aft_local)

        z=before_after_relation+before_local_relation+after_local_relation

        return z


class ConcatenationUnit_SecondSpatial(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(ConcatenationUnit_SecondSpatial, self).__init__()
        self.multi_relation_unit = RelationBetweenMulti(unit_nums, in_channels)
        self.pair_relation_unit = RelationBetweenPair(unit_nums, in_channels)
        self.spatial_relation = RelationOnSpatial(unit_nums, in_channels)
 
    def forward(self, x,y):   #return (n,128)
        origin_feature, local_feature, global_feature = x

        origin_relation = self.multi_relation_unit(y)
        local_relation = self.pair_relation_unit(y, local_feature)
        global_relation = self.spatial_relation(y, global_feature)
        
        z = origin_relation + local_relation + global_relation

        return z
    
class ConcatenationUnit_FirstTemporal(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(ConcatenationUnit_FirstTemporal, self).__init__()

        self.before_after = BeforeAfterFeature(unit_nums, in_channels)
        self.before_after_local = RelationBetweenPair(unit_nums, in_channels)

    def forward(self, x,x_bef,x_aft,x_bef_l,x_aft_l):   #return (n,128)

        origin_feature, local_feature, global_feature = x
        origin_feature_bef_loacl, local_feature_bef_local, global_feature_bef_local = x_bef_l
        origin_feature_aft_local, local_feature_aft_local, global_feature_aft_local = x_aft_l
        origin_feature_bef, local_feature_bef, global_feature_bef = x_bef
        origin_feature_aft, local_feature_aft, global_feature_aft = x_aft

        before_after_relation = self.before_after(origin_feature,origin_feature_bef,origin_feature_aft)
        before_local_relation = self.before_after_local(origin_feature,origin_feature_bef_loacl)
        after_local_relation = self.before_after_local(origin_feature,origin_feature_aft_local)

        z=before_after_relation+before_local_relation+after_local_relation

        return z

@RELATION_MODULE_REGISTER.register()
class RelationModule(nn.Module):
    def __init__(self, cfg, in_channels):
        super(RelationModule, self).__init__()
        self.relation_units = []
        unit_nums = cfg.MODEL.RELATION_HEAD.Relation_Unit_Nums
        for idx in range(unit_nums):
            relation_unit = 'relation_unit{}'.format(idx)
            #print(relation_unit)
            relation_unit_module = ConcatenationUnit(unit_nums, in_channels)
            self.add_module(relation_unit, relation_unit_module)
            self.relation_units.append(relation_unit)
        self.fc = Linear(in_channels, in_channels)
        nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, person_probs, person_features,
                x_bef, person_probs_bef, person_features_bef,
                x_aft, person_probs_aft, person_features_aft,
                x_bef_loacl,x_after_local):
        
        z = []
        #print(len(x))
        for x_, person_prob_per_img, person_features_per_img,x_b,x_a,x_b_l,x_a_l in zip(x, person_probs, person_features,x_bef,x_aft,x_bef_loacl,x_after_local):
            #print(len(x_))
            #print(type(x_[0]))
            #exit()
            result = tuple([getattr(self, relation_uint)(x_, person_prob_per_img, person_features_per_img,x_b,x_a,x_b_l,x_a_l)
                            for relation_uint in self.relation_units])   
            #print(len(result))
            y = torch.cat(result, dim=1)
            #print(y.shape)
            y = self.fc(y)
            z.append(F.relu(x_[0] + y))
        return z


@RELATION_MODULE_REGISTER.register()
class SpatialTemporal(nn.Module):
    def __init__(self, cfg, in_channels):
        super(SpatialTemporal, self).__init__()
        self.relation_units1 = []
        self.relation_units2 = []
        unit_nums = cfg.MODEL.RELATION_HEAD.Relation_Unit_Nums
        for idx in range(unit_nums):
            relation_unit1 = 'relation_unit1{}'.format(idx)
            relation_unit_module1 = ConcatenationUnit_FirstSpatial(unit_nums, in_channels)
            self.add_module(relation_unit1, relation_unit_module1)
            self.relation_units1.append(relation_unit1)

        for idx in range(unit_nums):
            relation_unit2 = 'relation_unit2{}'.format(idx)
            relation_unit_module2 = ConcatenationUnit_SecondTemporal(unit_nums, in_channels)
            self.add_module(relation_unit2, relation_unit_module2)
            self.relation_units2.append(relation_unit2)

        self.fc = Linear(in_channels, in_channels)
        nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, person_probs, person_features,
                x_bef, person_probs_bef, person_features_bef,
                x_aft, person_probs_aft, person_features_aft,
                x_bef_loacl,x_after_local):
        
        z = []
        for x_, person_prob_per_img, person_features_per_img,x_b,x_a,x_b_l,x_a_l in zip(x, person_probs, person_features,x_bef,x_aft,x_bef_loacl,x_after_local):

            result = tuple([getattr(self, relation_uint)(x_,x_b,x_a,'cur') for relation_uint in self.relation_units1])
            result_bef = tuple([getattr(self, relation_uint)(x_,x_b,x_a,'bef') for relation_uint in self.relation_units1])
            result_aft = tuple([getattr(self, relation_uint)(x_,x_b,x_a,'aft') for relation_uint in self.relation_units1])
            #print(result_bef)
            y_bef = torch.cat(result_bef, dim=1)
            y = torch.cat(result, dim=1)
            y_aft = torch.cat(result_aft, dim=1)

            result = tuple([getattr(self, relation_uint)(x_,y_bef,y,y_aft,x_b_l,x_a_l)
                                        for relation_uint in self.relation_units2])
            y = torch.cat(result, dim=1)
            y = self.fc(y)
            z.append(F.relu(x_[0] + y))

        #print(len(z))                     #batch
        # print(z[0])
        # print(z[0].shape)
        # exit()
        return z


@RELATION_MODULE_REGISTER.register()
class TemporalSpatial(nn.Module):
    def __init__(self, cfg, in_channels):
        super(TemporalSpatial, self).__init__()
        self.relation_units1 = []
        self.relation_units2 = []
        unit_nums = cfg.MODEL.RELATION_HEAD.Relation_Unit_Nums
        for idx in range(unit_nums):
            relation_unit1 = 'relation_unit1{}'.format(idx)
            relation_unit_module1 = ConcatenationUnit_FirstTemporal(unit_nums, in_channels)
            self.add_module(relation_unit1, relation_unit_module1)
            self.relation_units1.append(relation_unit1)

        for idx in range(unit_nums):
            relation_unit2 = 'relation_unit2{}'.format(idx)
            relation_unit_module2 = ConcatenationUnit_SecondSpatial(unit_nums, in_channels)
            self.add_module(relation_unit2, relation_unit_module2)
            self.relation_units2.append(relation_unit2)

        self.fc = Linear(in_channels, in_channels)
        nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, person_probs, person_features,
                x_bef, person_probs_bef, person_features_bef,
                x_aft, person_probs_aft, person_features_aft,
                x_bef_loacl,x_after_local):

        z = []
        #print(len(x))
        for x_, person_prob_per_img, person_features_per_img,x_b,x_a,x_b_l,x_a_l in zip(x, person_probs, person_features,x_bef,x_aft,x_bef_loacl,x_after_local):

            result = tuple([getattr(self, relation_uint)(x_,x_b,x_a,x_b_l,x_a_l) for relation_uint in self.relation_units1])

            y = torch.cat(result, dim=1)

            result = tuple([getattr(self, relation_uint)(x_,y)
                                        for relation_uint in self.relation_units2])
            y = torch.cat(result, dim=1)
            y = self.fc(y)
            z.append(F.relu(x_[0] + y))

        return z

def make_relation_module(cfg, in_channels, method):
    relation_module = RELATION_MODULE_REGISTER.get(method)(cfg, in_channels)
    return relation_module
