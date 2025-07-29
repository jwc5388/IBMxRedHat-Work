import torch
import torch.version

print('Pytorch version:', torch.version)

cuda_available= torch.cuda.is_available()
print('CUDA 사용 가능 여부:', cuda_available)

gpu_count = torch.cuda.device_count()
print('사용 가능 gpu 갯수:', gpu_count )

if cuda_available:
    current_device = torch.cuda.current_device()
    print('현재 사용중인 gpu 장치 ID:', current_device)
    print('현재 gpu 이름:', torch.cuda.get_device_name(current_device))
else: 
    print('gpu 없음')
    
    
print('CUDA 버전 :', torch.version.cuda)

cudnn_version = torch.backends.cudnn.version()
if cudnn_version is not None:
    print('cuDNN version:', cudnn_version)
else:
    print('cuDNN 없음')