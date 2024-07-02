# -*-coding:utf-8-*-
import sys
import os
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import torch
import onnx
import onnxruntime
import numpy as np
import torch.nn as nn
from ltr.models.tracking.hcat import MLP
from ltr.models.backbone.transt_backbone import build_backbone,Backbone
from ltr.models.neck.featurefusion_network import build_featurefusion_network
import ltr.admin.settings as ws_settings
from ltr.models.neck.position_encoding import build_position_encoding
from thop import profile
from thop.utils import clever_format
import copy
import time
import numpy as np
import onnxruntime as ort
from onnxconverter_common import float16
import onnx
from onnxsim import simplify as simplify_func
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class HCAT(nn.Module):
    def __init__(self, backbone, featurefusion_network, num_classes):
        super().__init__()
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(16, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, search,feature_template,pos_template):
        feature_search, pos_search = self.backbone(search)
        src_search = feature_search[-1]
        src_template = feature_template
        hs = self.featurefusion_network(self.input_proj(src_template), self.input_proj(src_search), pos_template, pos_search[-1], self.query_embed.weight)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out


def convert_tracking_model(net,search_size,feature_size_template,feature_template_d):
    x = torch.randn(1, 3, search_size, search_size).cuda()
    feature_template = torch.randn(1, feature_template_d, feature_size_template, feature_size_template).cuda()
    pos_template = torch.randn(1, 256, feature_size_template, feature_size_template).cuda()
    ort_inputs = {'x': to_numpy(x).astype(np.float32),
                  'feature_template':to_numpy(feature_template).astype(np.float32),
                  'pos_template':to_numpy(pos_template).astype(np.float32)}
    ###########complete model pytorch->onnx#############
    print("Converting tracking model now!")
    torch.onnx.export(net, (x,feature_template,pos_template), 'models/com_cov.onnx_test', export_params=True,
          opset_version=11, do_constant_folding=True, input_names=['x','feature_template','pos_template'],
          output_names=['cls','reg'])
    #######load the converted model and inference#######
    with torch.no_grad():
        oup = net(x,feature_template,pos_template)
        onnx_model = onnx.load("models/com_cov.onnx_test")
        onnx.checker.check_model(onnx_model)
        ort_session = onnxruntime.InferenceSession("models/com_cov.onnx_test")
        ort_outs = ort_session.run(None, ort_inputs)
        np.testing.assert_allclose(to_numpy(oup['pred_logits']), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("The deviation between the first output: {}".format(np.max(np.abs(to_numpy(oup['pred_logits'])-ort_outs[0]))))
        print("The deviation between the second output: {}".format(np.max(np.abs(to_numpy(oup['pred_boxes'])-ort_outs[1]))))
    print(onnxruntime.get_device())
    print("Tracking onnx model has done!")

def convert_template_model(net,template_size):
    zf = torch.randn(1, 3, template_size, template_size).cuda()
    onnx_inputs = {'zf': to_numpy(zf).astype(np.float32)}
    backbone = net.backbone
    print("Converting template model now!")
    torch.onnx.export(backbone, (zf), 'models/back_cov.onnx_test', export_params=True,
          opset_version=11, do_constant_folding=True, input_names=['zf'],
          output_names=['out','pos'])
    with torch.no_grad():
        out,pos = backbone(zf)
        onnx_backbone = onnx.load("models/back_cov.onnx_test")
        onnx.checker.check_model(onnx_backbone)
        ort_session = onnxruntime.InferenceSession("models/back_cov.onnx_test")
        ort_outs = ort_session.run(None, onnx_inputs)
        np.testing.assert_allclose(to_numpy(pos[0]), ort_outs[1], rtol=1e-03, atol=1e-05)
        print("The deviation between the first output: {}".format(np.max(np.abs(to_numpy(out[0])-ort_outs[0]))))
        print("The deviation between the second output: {}".format(np.max(np.abs(to_numpy(pos[0])-ort_outs[1]))))
    print(onnxruntime.get_device())
    print("Template onnx model has done!")

if __name__ == "__main__":
    # convert model to onnx
    settings = ws_settings.Settings()
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256

    settings.backbone = 'convnext_tiny'
    # settings.backbone = 'resnet50'
    # settings.backbone = 'resnet18'
    # settings.backbone = 'lightrack_backbone_M'
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048
    settings.featurefusion_layers = 1
    search_size = 256
    template_size = 128
    feature_size_template = 8   #stride16
    if settings.backbone == 'convnext_tiny':
        feature_template_d = 384
    elif settings.backbone == 'resnet50':
        feature_template_d = 1024
    elif settings.backbone == 'resnet18':
        feature_template_d = 256
    else:
        feature_template_d = 96
    use_gpu = False
    num_classes = 1
    model_path = os.path.join(os.path.dirname(__file__),'models','convnext_tiny_N2_q16.pth')
    backbone_net = build_backbone(settings,backbone_pretrained=False)
    featurefusion_network = build_featurefusion_network(settings)
    net = HCAT(backbone_net,featurefusion_network,num_classes=num_classes)
    # checkpoints = torch.load(model_path)['net']
    # net.load_state_dict(torch.load(model_path)['net'],strict=False)
    net.eval()
    if use_gpu:
        net.cuda()
        device = 'cuda:0'
    else:
        device = 'cpu'

    ### INPUT
    search = torch.randn(1, 3, search_size, search_size).to(device)
    # template = torch.randn(1, 3, template_size, template_size).cuda()
    feature_template = torch.randn(1, feature_template_d, feature_size_template, feature_size_template).to(device)
    pos_template = torch.randn(1, 256, feature_size_template, feature_size_template).to(device)

    dtype = np.float32
    inputs=(search, feature_template, pos_template)
    inputs_onnx = {'x':  np.array(search.cpu(), dtype=dtype),
                   'zf': np.array(feature_template.cpu(), dtype=dtype),
                   'zp': np.array(pos_template.cpu(), dtype=dtype),
                #    'tbb': np.array(template_bb.cpu(), dtype=dtype),
                   }

    model = net
    model_ = copy.deepcopy(model)
    macs1, params1 = profile(model, inputs=(search, feature_template, pos_template),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    torch.onnx.export(model, 
        inputs,
        'test_net.onnx', 
        input_names=[ "x", "zf", 'zp'], 
        output_names=["output"],
        opset_version=11,
        export_params=True,
        # verbose=True,
        # dynamic_axes={'input':{0:'batch', 2:'h', 3:'w'}, 'output':{0:'batch', 2:'h2', 3:'w2'}} 
    )

    providers = ['CPUExecutionProvider']
    ort_session = ort.InferenceSession("test_net.onnx", providers=providers)

    T_w = 100
    T_t = 500
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            # _ = model_(search, feature_template, pos_template)
            output = ort_session.run(output_names=['output'],
                             	input_feed=inputs_onnx,
                                )
        start = time.time()
        for i in range(T_t):
            # _ = model_(search, feature_template, pos_template)
            output = ort_session.run(output_names=['output'],
                             	input_feed=inputs_onnx,
                                )
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))
    ######convert and check tracking pytorch model to onnx#####
    # convert_tracking_model(net,search_size,feature_size_template,feature_template_d)
    ######convert and check template pytorch model to onnx#####
    # convert_template_model(net,template_size)

