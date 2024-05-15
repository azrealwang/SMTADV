# Targeted attack to face recognition system
import eagerpy as ep
import codecs
import time
import torch
from math import ceil
from torch import Tensor
from scipy.io import loadmat
from facenet_pytorch import InceptionResnetV1
from insightface import iresnet100
import foolbox.attacks as attacks
from foolbox.models import PyTorchModel
from foolbox.utils import samples, FMR, cos_similarity_score
import pytorch_ssim
import lpips
from torchvision.utils import save_image
from scipy.io import savemat

def main() -> None:
    # Settings
    samplesize = int(10)
    subject = int(2)
    foldersize = int(samplesize*subject/2)
    source = "lfw" # lfw, CelebA-HQ
    target = "CelebA-HQ" # lfw, CelebA-HQ
    dfr_model1 = 'facenet'
    dfr_model2 = 'insightface'
    threshold1 = 0.7032619898135847 # facenet
    threshold2 = 0.5854403972629942 # insightface
    attack_model = attacks.LinfPGD
    loss_type = 'MS' #'ST', 'MT', 'MS', 'C-BCE'
    epsilons = 0.03
    steps = 1000
    step_size = 0.001
    convergence_threshold = 0.0001
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    totalsize = samplesize*subject
    batchsize = 30

    # Log
    log_time = time.strftime("%Y%m%d%H%M%S",time.localtime())
    f = codecs.open(f'results/logs/{loss_type}_{source}_{target}_{attack_model.__module__}_{log_time}.txt','a','utf-8')
    f.write(f"samplesize = {samplesize}, subject = {subject}, source = {source}, target = {target}, dfr_model1 = {dfr_model1}, dfr_model2 = {dfr_model2}, threshold1 = {threshold1}, threshold2 = {threshold2}\n")
    f.write(f"attack_model = {attack_model}, loss_type = {loss_type}, epsilons = {epsilons}, steps = {steps}, step_size = {step_size}, convergence_threshold = {convergence_threshold}, batchsize = {batchsize}\n")
    
    # Model
    model1 = InceptionResnetV1(pretrained='vggface2').eval()
    model2 = iresnet100(pretrained=True).eval()
    mean=[0.5]*3
    std=[0.5]*3
    bounds=(0, 1)
    preprocessing = dict(mean = mean, std = std, axis=-3)
    fmodel1 = PyTorchModel(model1, bounds=bounds, preprocessing=preprocessing)
    fmodel2 = PyTorchModel(model2, bounds=bounds, preprocessing=preprocessing)
    
    # Load data
    features_tmp1 = loadmat(f'mat/{target}_{dfr_model1}_templates.mat')[f'{target}_{dfr_model1}_templates']
    features1 = Tensor(features_tmp1)
    features_tmp2 = loadmat(f'mat/{target}_{dfr_model2}_templates.mat')[f'{target}_{dfr_model2}_templates']
    features2 = Tensor(features_tmp2)
    source_images, _ = ep.astensors(*samples(fmodel1, dataset=f"{source}_test", batchsize=subject*samplesize, model_type=dfr_model2))
    
    # Input data
    attack_index = list(range(samplesize*subject))
    attack_images = source_images[attack_index]
    target_index = list(range(foldersize,foldersize*2))+list(range(samplesize,foldersize))+list(range(0,samplesize))
    target_features1 = features1[target_index]
    target_features2 = features2[target_index]
    del source_images
    
    # Run attack
    attack = attack_model(loss_type=loss_type, steps=steps, abs_stepsize=step_size, convergence_threshold=convergence_threshold, device=device, threshold=threshold1, threshold2=threshold2)
    raw_advs = Tensor([]).to(device)
    advs_features1 = Tensor([]).to(device)
    advs_features2 = Tensor([]).to(device)
    time_cost = 0
    for i in range(ceil(totalsize/batchsize)):
        print(f"Batch: {i+1}")
        start = i*batchsize
        if i == ceil(totalsize/batchsize)-1:
            batchsize = totalsize - batchsize*i
        start_time = time.time()
        raw_advs_tmp, _, _ = attack(fmodel1, attack_images[start:start+batchsize], target_features1[start:start+batchsize],criterion2=target_features2[start:start+batchsize], model2=fmodel2, epsilons=epsilons)
        end_time = time.time()
        time_cost = time_cost + end_time - start_time
        advs_features1_tmp = fmodel1(raw_advs_tmp)
        advs_features2_tmp = fmodel2(raw_advs_tmp)
        raw_advs = torch.cat((raw_advs, raw_advs_tmp.raw),0)
        advs_features1 = torch.cat((advs_features1, advs_features1_tmp.raw),0)
        advs_features2 = torch.cat((advs_features2, advs_features2_tmp.raw),0)
        del raw_advs_tmp, advs_features1_tmp, advs_features2_tmp
    del attack, fmodel1, model1, fmodel2, model2
    print(f"Attack costs {time_cost}s")
    f.write(f"Attack costs {time_cost}s\n")

    # Save advs template
    adv_template1 = advs_features1.cpu().numpy()
    savemat(f'mat/{loss_type}_{source}_{target}_{dfr_model1}_templates.mat', mdict={f"{loss_type}_{source}_{target}_{dfr_model1}_templates": adv_template1})
    adv_template2 = advs_features2.cpu().numpy()
    savemat(f'mat/{loss_type}_{source}_{target}_{dfr_model2}_templates.mat', mdict={f"{loss_type}_{source}_{target}_{dfr_model2}_templates": adv_template2})
    
    # Save advs
    save_image(raw_advs[0], f'results/images/{loss_type}_{source}_{target}_{log_time}_adv.jpg')
    noise = (raw_advs[0]-attack_images[0].raw+bounds[1]-bounds[0])/((bounds[1]-bounds[0])*2)
    save_image(noise, f'results/images/{loss_type}_{source}_{target}_{log_time}_noise.jpg')
    del noise
    
    # Compute SSIM
    ssim_loss = pytorch_ssim.SSIM()
    ssim_score = ssim_loss(attack_images.raw,raw_advs)
    print(f"SSIM = {ssim_score}")
    f.write(f"SSIM = {ssim_score}\n")
    del ssim_loss, ssim_score
    
    # Compute LPIPS
    loss_fn = lpips.LPIPS(net='alex')
    if bounds != (-1, 1):
        attack_images = attack_images.raw.cpu()*2-1
        raw_advs = raw_advs.cpu()*2-1
    lpips_score = loss_fn.forward(attack_images,raw_advs).mean()
    print(f"LPISP = {lpips_score}")
    f.write(f"LPISP = {lpips_score}\n")
    del attack_images, raw_advs, loss_fn
    
    #Compute dissimilarity
    dissimilarity1 = 1-cos_similarity_score(advs_features1,target_features1).mean()
    dissimilarity2 = 1-cos_similarity_score(advs_features2,target_features2).mean()
    print(f"Dissimilarity = {dissimilarity1}, {dissimilarity2}")
    f.write(f"Dissimilarity = {dissimilarity1}, {dissimilarity2}\n")
    
    # Compute FMR
    fmr_target1, fmr_renew1 = FMR(advs_features1, target_features1, threshold1, samplesize)
    fmr_target2, fmr_renew2 = FMR(advs_features2, target_features2, threshold2, samplesize)
    print("Attack performance:")
    f.write("Attack performance:\n")
    print(f" advs vs targets: FMR = {fmr_target1 * 100:.2f}%, {fmr_target2 * 100:.2f}%")
    f.write(f" advs vs targets: FAR = {fmr_target1 * 100:.2f}%, {fmr_target2 * 100:.2f}%\n")
    print(f" advs vs renews: FMR = {fmr_renew1 * 100:.2f}%, {fmr_renew2 * 100:.2f}%")
    f.write(f" advs vs renews: FAR = {fmr_renew1 * 100:.2f}%, {fmr_renew2 * 100:.2f}%\n")

    f.close()

if __name__ == "__main__":
    main()
