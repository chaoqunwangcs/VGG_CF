# VGG_CF
Visual Object Tracking based on Correlation Filter using VGG feature. About 2s per frame, can be faster via optimize the code.  
Precision rate:86.3%, Success rate: 60.3%

# Dependences
CUDA8.0  
cudnn6.0  
python==3.6  
GPU ~=4.5G memory

# Tracking
1. git clone https://github.com/wangchaoqun56/VGG_CF.git
2. pip install -r requirments.txt
3. cd tracking  
	eidt configer.py  
	data_path: path to dataset  
	vgg_model_path: path to vgg19 model pretrained by ImageNet
4. python tracker.py -s 0 -e 100  
	--start: index of first sequence  
	--end: index of last sequence  
	--gpu: gpu id default='0'

# Other
download vgg19.npy from [vgg19.npy](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)