import pickle
import numpy as np
import paddle as pdl
import cv2 as cv

yN = 14

COLS = [
	'Impression',
	'HyperF_Type',
	'HyperF_Area(DA)',
	'HyperF_Fovea',
	'HyperF_ExtraFovea',
	'HyperF_Y',
	'HypoF_Type',
	'HypoF_Area(DA)',
	'HypoF_Fovea',
	'HypoF_ExtraFovea',
	'HypoF_Y',
	'CNV',
	'Vascular abnormality (DR)',
	'Pattern'
]

Mcols = [
	'Impression',
	'HyperF_ExtraFovea',
	'HyperF_Y',
	'HypoF_ExtraFovea',
	'HypoF_Y',
	'Vascular abnormality (DR)',
	'Pattern',
]

McolsN = {
	'Impression': 8,
	'HyperF_ExtraFovea': 8,
	'HyperF_Y': 5,
	'HypoF_ExtraFovea': 8,
	'HypoF_Y': 5,
	'Vascular abnormality (DR)': 7,
	'Pattern': 7,
}

_k2v = {
	'Impression': {
		'macular neovascularization': 0,
		'unremarkable changes': 1,
		'dry age-related macular degeneration': 2,
		'central serous chorioretinopathy': 3,
		'uveitis': 4,
		'chorioretinal scar': 5,
		'diabetic retinopathy': 6,
		'retinal pigment epithelial detachment': 7,
		'polypoidal choroidal vasculopathy': 8,
		'pachychoroid pigment epitheliopathy': 9,
		'chorioretinal atrophy': 10,
		'myopia': 11,
		'cystoid macular edema': 12,
		'choroidal mass': 13,
		'epiretinal membrane': 14,
		'retinal vein occlusion': 15,
		'other': 16,
		'retinal arterial macroaneurysm': 17,
		'proliferative diabetic retinopathy': 18,
		'branch retinal vein occlusion': 19,
		'retinal dystrophy': 20,
		'central retinal vein occlusion': 21,
		'central retinal artery occlusion': 22
	},
	'HyperF_Type': {
		'leakage': 0,
		'no': 1,
		'staining': 2,
		'pooling': 3,
		'window defect': 4
	},
	'HyperF_Area(DA)': {
		'4': 0,
		'no': 1,
		'5': 2
	},
	'HyperF_Fovea': {
		'no': 0,
		'yes': 1
	},
	'HyperF_ExtraFovea': {
		'no': 0,
		'superior': 1,
		'diffuse': 2,
		'temporal': 3,
		'inferior': 4,
		'nasal': 5,
		'superior nasal': 6,
		'superior temporal': 7,
		'inferior temporal': 8,
		'inferior nasal': 9,
		'disc': 10,
		'superior to disc': 11,
		'nasal to disc': 12,
		'inferior to disc': 13,
		'temporal to disc': 14,
		'periphery': 15,
		'venular': 16,
		'superotemporal': 17
	},
	'HyperF_Y': {
		'subretinal': 0,
		'no': 1,
		'intraretinal': 2,
		'preretinal': 3
	},
	'HypoF_Type': {
		'no': 0,
		'blockage': 1,
		'capillary non-perfusion': 2
	},
	'HypoF_Area(DA)': {
		'no': 0,
		'4': 1,
		'5': 2
	},
	'HypoF_Fovea': {
		'no': 0,
		'yes': 1
	},
	'HypoF_ExtraFovea': {
		'no': 0,
		'inferior': 1,
		'diffuse': 2,
		'superior': 3,
		'temporal': 4,
		'nasal': 5,
		'inferior temporal': 6,
		'superior nasal': 7,
		'inferior nasal': 8,
		'superior temporal': 9,
		'superior to disc': 10,
		'disc': 11,
		'nan': 12,
		'temporal to disc': 13,
		'periphery': 14,
		'inferior to disc': 15,
		'nasal to disc': 16
	},
	'HypoF_Y': {
		'no': 0,
		'subretinal': 1,
		'intraretinal': 2,
		'preretinal': 3,
		'nan': 4
	},
	'CNV': {
		'no': 0,
		'yes': 1
	},
	'Vascular abnormality (DR)': {
		'no': 0,
		'microaneurysm': 1,
		'tortuous': 2,
		'intraretinal microvascular abnormalities': 3,
		'retinal neovascularization elsewhere': 4,
		'macroaneurysm': 5,
		'vessel dilation': 6,
		'telangiectasia': 7,
		'collateral vessel': 8,
		'venous beading': 9,
		'retinal neovascularization of the disc': 10,
		'optociliary shunt': 11,
		'vasculitis': 12,
		'mild tortuous vessel': 13,
		'tortuous dilate': 14
	},
	'Pattern': {
		'no': 0,
		'pcv': 1,
		'polyp': 2,
		'ink blot': 3,
		'petaloid': 4,
		'drusen': 5,
		'smoke stack': 6,
		'panretinal photocoagulation': 7,
		'light bulb': 8,
		'starry sky': 9,
		'retinal pigment epithelial tear': 10,
		'laser scar': 11,
		'branching neovascular network': 12,
		'segmental panretinal photocoagulation': 13
	}
}

_k2v_num = {
	_k1: len(_v1)
	for _k1, _v1 in _k2v.items()
}

_v2k = {
	_k1: {_v2: _k2 for _k2, _v2 in _v1.items()}
	for _k1, _v1 in _k2v.items()
}

#
fnormalize = lambda x, _mean=70.132575, _std=34.550730: (x - _mean) / _std
fill32 = lambda x: np.array([1 if i == x else 0 for i in range(32)])
_x2N = 11

class iDataset(pdl.io.Dataset):
	def __init__(self, data_name, data_list):
		self.data_name = data_name
		self.data_list = data_list
		self.D = 40
		self.L, self.R = 384//2, 384 + 384//2

	def __getitem__(self, index):
		_data = self.data_list[index]
		
		img0 = _data["X"]
		img_b, img_g, img_r = cv.split(img0)
		img_m1 = img0[self.L-self.D:self.L+self.D, self.L-self.D:self.L+self.D, ]
		img_m2 = img0[self.L-self.D:self.L+self.D, self.R-self.D:self.R+self.D, ]

		_x1 = fnormalize(_data["X"]).reshape((3, 768, 384)).astype(np.float32)
		_x2 = np.array([
			img0.max(),
			np.percentile(img0, 10),
			img_b.max(),
			img_g.max(),
			img_r.max(),

			img_m1.mean(), img_m1.max(), img_m1.min(),
			img_m2.mean(), img_m2.max(), img_m2.min(),
		], dtype=np.float32)
		
		_y = np.concatenate([
			fill32(_data["Y"]["Impression"]),
			fill32(_data["Y"]["HyperF_Type"]),
			fill32(_data["Y"]["HyperF_Area(DA)"]),
			fill32(_data["Y"]["HyperF_Fovea"]),
			fill32(_data["Y"]["HyperF_ExtraFovea"]),
			fill32(_data["Y"]["HyperF_Y"]),
			fill32(_data["Y"]["HypoF_Type"]),
			fill32(_data["Y"]["HypoF_Area(DA)"]),
			fill32(_data["Y"]["HypoF_Fovea"]),
			fill32(_data["Y"]["HypoF_ExtraFovea"]),
			fill32(_data["Y"]["HypoF_Y"]),
			fill32(_data["Y"]["CNV"]),
			fill32(_data["Y"]["Vascular abnormality (DR)"]),
			fill32(_data["Y"]["Pattern"]),
		], dtype=np.float32)
		return _x1, _x2, _y

	def __len__(self):
		return len(self.data_list)

def jDataset(_data):
	selfD = 40
	selfL, selfR = 384//2, 384 + 384//2

	img0 = _data
	img_b, img_g, img_r = cv.split(img0)
	img_m1 = img0[selfL-selfD:selfL+selfD, selfL-selfD:selfL+selfD, ]
	img_m2 = img0[selfL-selfD:selfL+selfD, selfR-selfD:selfR+selfD, ]

	_x1 = fnormalize(_data).reshape((3, 768, 384)).astype(np.float32)
	_x2 = np.array([
		img0.max(),
		np.percentile(img0, 10),
		img_b.max(),
		img_g.max(),
		img_r.max(),

		img_m1.mean(), img_m1.max(), img_m1.min(),
		img_m2.mean(), img_m2.max(), img_m2.min(),
	], dtype=np.float32)
	return pdl.Tensor(np.array([_x1])), pdl.Tensor(np.array([_x2]))

def model():
	return pdl.nn.Sequential(
		pdl.nn.Linear(in_features=512, out_features=256),
		pdl.nn.ReLU(),

		pdl.nn.Linear(in_features=256, out_features=128),
		pdl.nn.ReLU(),

		pdl.nn.Linear(in_features=128, out_features=64),
		pdl.nn.ReLU(),

		pdl.nn.Linear(in_features=64, out_features=32),
		pdl.nn.ReLU(),

		pdl.nn.Linear(in_features=32, out_features=32),
		pdl.nn.ReLU(),

		pdl.nn.Linear(in_features=32, out_features=32),
	)


class RES(pdl.nn.Layer):
	def __init__(self):
		super(RES, self).__init__()
		self.resnet = pdl.nn.Sequential(
			pdl.vision.models.resnet152(num_classes=512, pretrained=True)
		)

		self.model1 = model()
		self.model2 = model()
		self.model3 = model()
		self.model4 = model()
		self.model5 = model()
		self.model6 = model()
		self.model7 = model()
		self.model8 = model()
		self.model9 = model()
		self.modelA = model()
		self.modelB = model()
		self.modelC = model()
		self.modelD = model()
		self.modelE = model()

	def forward(self, _x):
		_res = self.resnet(_x)
		return pdl.concat([
			self.model1(_res),
			self.model2(_res),
			self.model3(_res),
			self.model4(_res),
			self.model5(_res),
			self.model6(_res),
			self.model7(_res),
			self.model8(_res),
			self.model9(_res),
			self.modelA(_res),
			self.modelB(_res),
			self.modelC(_res),
			self.modelD(_res),
			self.modelE(_res),
		], axis=1)

class PaiPai(pdl.nn.Layer):
	def __init__(self, _N=32 * yN):
		super(PaiPai, self).__init__()
		self._N = _N

		self.model1 = RES()
		self.model1.set_state_dict(pdl.load("./modl/_s4.9997_t5.6384_v4.9997.mdl"))

		self.model2 = pdl.nn.Sequential(
			pdl.nn.Linear(in_features=_x2N, out_features=self._N),
			pdl.nn.ReLU(),

			pdl.nn.Linear(in_features=self._N, out_features=self._N),
			pdl.nn.ReLU(),

			pdl.nn.Linear(in_features=self._N, out_features=self._N),
		)

		self.b0 = self.create_parameter(
			[1], is_bias=True, default_initializer=pdl.nn.initializer.Constant(value=0.0))
		self.w1 = self.create_parameter(
			[1], is_bias=True, default_initializer=pdl.nn.initializer.Constant(value=1.0))
		self.w2 = self.create_parameter(
			[1], is_bias=True, default_initializer=pdl.nn.initializer.Constant(value=0.0))

	def forward(self, _x1, _x2):
		return self.b0 + self.w1 * self.model1(_x1) + self.w2 * self.model2(_x2)


iMessages = lambda _i, _k: [
	{
		"role": "user",
		"content": "你是一个眼科医生",
	}, 
	{
		"role": "assistant",
		"content": "是的，我现在扮演的是眼科医生的角色。根据您提供的眼部信息，我可以给您一些建议，帮助您更好地了解您的眼部健康状况。",
	}, 

	{
		"role": "user",
		"content": "我们的诊断信息主要来自于眼底萤光血管造影，它是一种侵入式的检查，是由手臂静脉注入萤光剂，再由眼底相机照出黄斑部及视网膜血管的变异情形，可作为医师诊断及后续治疗的参考辅助。",
	}, 
	{
		"role": "assistant",
		"content": "具体来说，诊断信息包含哪些内容，我可以知道么？",
	}, 

	{
		"role": "user",
		"content": """
			Data of each eye contains the following information
			每只眼睛的治疗前数据包含以下信息

			ID: patient ID
			
			Folder
			文件夹(眼睛)
			
			Impression of each eye
			每只眼睛的诊断结果
			
			HyperF_Type: hyperfluorescence type
			高荧光类型

			HyperF_Area (DA)
			-4 (The lesion is 4 or smaller than 4 disc areas)
			≤ 4个视盘面积
			-5 (The lesion is larger than 4 disc areas)
			＞4个视盘面积

			HyperF_Fovea: hyperfluorescence (fovea)
			黄斑区高荧光
			
			HyperF_ExtraFovea: hyperfluorescence (extra fovea)
			黄斑外高荧光
			
			HyperF_Y: hyperfluorescence (Y location)
			高荧光纵轴位置
			
			HypoF_Type: hypofluorescence type
			低荧光类
			
			HypoF_Area (DA)
			-4 (The lesion is 4 or smaller than 4 disc areas)
			≤ 4个视盘面积
			-5 (The lesion is larger than 4 disc areas)
			＞4个视盘面积
			
			HypoF_Fovea: hypofluorescence (fovea)
			黄斑区低荧光
			
			HypoF_ExtraFovea: hypofluorescence (extra fovea)
			黄斑外低荧光
			
			HypoF_Y: hypofluorescence (Y location)
			低荧光纵轴位置
			
			CNV: choroidal neovascularization
			脉络膜新生血管
			
			Vascular abnormality (DR)
			血管异常（糖尿病视网膜病变特征）
			
			Pattern
			特殊荧光特殊类型

			Angiography images: including FFA (fundus fluorescein angiography) and/or ICGA (indocyanine green angiography) images. There are multiple images taken based on examination needs.
			造影图像：包括眼底荧光血管造影和/或吲哚菁绿血管造影
		""",
	}, 
	{
		"role": "assistant",
		"content": "好的，我知道了。",
	}, 

    # # 纯文本
	# {
	# 	"role": "user",
	# 	"content": f"{_k} 请根据这些内容，整合成一份眼科疾病诊断报告",
	# },

    # 多模态
    {
		"role": "user",
        "content": [
            {
                "type": "text", 
                "text": f"{_k} 请根据这些内容，整合成一份眼科疾病诊断报告"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_i}"
                }
            }
        ]
    	},
]

# 