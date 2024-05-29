import os
import numpy as np
import numpy.matlib
import scipy as sp
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"#Unreal data is in .EXR format which may trigger firewalls. This line allows .EXR files to be read correctly
import cv2
from matplotlib.cm import ScalarMappable
import time
from tqdm import tqdm
import random

plt.rcParams['figure.dpi'] = 150#Only for display purposes in JupyterNotebook

def calc_img(pix_y,pix_x,peak_po,sig):

    #This function adds to each pixels' depth a random value drawn from a normal distribution according to the Fisher information

    mu = np.zeros((pix_y,pix_x))

    off_set = np.random.normal(mu, sig, size = (pix_y,pix_x))

    spad_img = peak_po+off_set    

    return spad_img

def calc_liklyhood(num_frames,C_ref,bckg_refs,bins,exp_time,rep_rate,energy_per_pulse,peak_posi,pulse_width_fwhm,spad_q_efficiency,t_res,target_range,illum_radius,effictive_pix_size_x,effictive_pix_size_y,f_no,dark_count_rate,solar_background_per_meter):
    
    #这个函数用来计算每脉冲的光子数

    illum_area = np.pi*(illum_radius)**2 # 照明区域面积

    C_atm = 0.9 # 海水透射系数

    wavelength = 532e-9 # 激光器波长

    c_speed = 2.25e8 # 光速
    plank = 6.626e-34 # 普朗克常量

    sig_photon = ((wavelength*energy_per_pulse)/(plank*c_speed))*((spad_q_efficiency*C_ref*(C_atm**(2*target_range)))/(8*((f_no**2))))*((effictive_pix_size_x*effictive_pix_size_y)/(illum_area)) # SPAD每脉冲探测到的平均光子数

    dark_counts = dark_count_rate #The dark count rate of the detector

    solar_background_off_of_target_photon = ((wavelength*solar_background_per_meter)/(plank*c_speed))*((spad_q_efficiency*bckg_refs*(C_atm**(target_range)))/(8*((f_no**2))))*((effictive_pix_size_x*effictive_pix_size_y)/(illum_area))#The average number of solar background photons measured by the detector per laser pulse

    return sig_photon,solar_background_off_of_target_photon

def calc_cont_fisher_info(peak_posi,jitter,num_pulses,bins,t_res,solar_background_off_of_target_photon,dark_count_rate,sig_photon,pulse_width_fwhm,num_frames):
    
    # 这个方程用来计算fisher信息

    r = peak_posi.shape[0] # 图像行数
    c = peak_posi.shape[1] # 图像列数

    t_vec_samples = 5000 # 时间采样率
    t_vec = np.linspace(0,bins,t_vec_samples)*t_res # TCSPC window
    t_vec = np.tile(t_vec,(r,c,1)) # The TCSPC window tiled out to the image size

    sigma = pulse_width_fwhm/(2*np.sqrt(2*np.log(2))) # 脉冲响应方程的宽度 
    sigma = np.ones((r,c,t_vec_samples))*sigma # 扩展到矩阵
    

    sig_photon = np.ones((r,c,t_vec_samples))*sig_photon # 矩阵化的平均信号光子数
    solar_background_off_of_target_photon = np.ones((r,c,t_vec_samples))*solar_background_off_of_target_photon # 矩阵化的平均背景光子数
    dark_count_rate = np.ones((r,c,t_vec_samples))*dark_count_rate # 矩阵化的暗计数
    peak_posi = np.repeat(peak_posi[:,:,np.newaxis],t_vec_samples,axis=2)#The true depth value repeated along axis 2 to allow for array operations
    

    numer = ((sig_photon)**2)*((t_vec-peak_posi)**2)*np.exp(-((t_vec-peak_posi)/(sigma))**2) # The numerator of the fisher info equation

    alpha = (((solar_background_off_of_target_photon+dark_count_rate)*(t_res*bins))+sig_photon) # The alpha constant

    denom = ((sigma**6)*(2*np.pi*alpha))*(solar_background_off_of_target_photon+dark_count_rate+((sig_photon/(sigma*np.sqrt(2.0*np.pi)))*(np.exp(-0.5*((t_vec-peak_posi)/sigma)**2)))) # The denominator of the fisher info equation


    fisher_vec = numer/denom#The fisher information integrand for each pixel

    fisher_info = np.trapz(fisher_vec,t_vec,axis=2)#The fisher information for each pixel

    return fisher_info,alpha[:,:,0]

def make_imgs(bins,peak_posi,jitter,C_ref,bckg_refs,num_frames,f_num,num_imgs,pix_x,pix_y):
    #This function creates images using the fisher information sampling approach
    

    fps = 1000 # 相机帧率

    exp_time = 1/fps # 曝光时间

    rep_rate = 10e6 # 激光重频

    energy_per_pulse = 4e-9 # 单脉冲能量

    pulse_width_fwhm = 0.072e-9 # 脉宽

    t_res = 150e-12 # SPAD时间分辨率

    num_pulses = int(np.round_(exp_time*rep_rate)) # 单帧脉冲数
    
    spad_q_base_efficiency = 0.28 # SPAD量子效率

    spad_q_efficiency = spad_q_base_efficiency
        
    # target_range = 14.73 # 目标距离
    target_range = 13.00

    fibre_core = 550e-6 # 激光光斑直径
    illum_lens = 8e-3 # 物镜焦距

    illum_radius = ((target_range/illum_lens)*fibre_core)/2.0 # 照明区域半径

    effictive_pix_size_x = 9.2e-6 # 有效像素宽度
    effictive_pix_size_y = 9.2e-6 # 有效像素高度

    f_no = f_num # 物镜光圈数

    dark_count_rate = 126 # 暗计数率
    solar_background_per_meter = 0.0 # 背景太阳光

    sig_photon,solar_background_off_of_target_photon = calc_liklyhood(num_frames,C_ref,bckg_refs,bins,exp_time,rep_rate,energy_per_pulse,peak_posi,pulse_width_fwhm,spad_q_efficiency,t_res,target_range,illum_radius,effictive_pix_size_x,effictive_pix_size_y,f_no,dark_count_rate,solar_background_per_meter)#The average number of signal and background photons measured by the sensor per pulse

    cont_fisher_info,alpha = calc_cont_fisher_info(peak_posi,jitter,num_pulses,bins,t_res,solar_background_off_of_target_photon,dark_count_rate,sig_photon,pulse_width_fwhm,num_frames)#The Fisher information for each pixel

    photon_per_frame = 1-((1-alpha)**num_pulses) # 平均单帧光子数

    spc_ave = num_frames*(photon_per_frame) # 每个直方图的平均光子数

    spc_std = np.sqrt(num_frames*(photon_per_frame)*(1-photon_per_frame)) # 探测到光子的标准差

    spc = np.round_(np.random.normal(spc_ave, spc_std)) # 探测到的光子数

    sig = (((1/(np.sqrt(spc*np.asarray(cont_fisher_info))))))/np.asarray(t_res) # The Cramer-Rao bound

    peak_posi = np.asarray(peak_posi/t_res) # The peak positions converted to bin index

    spad_com_img = np.zeros((pix_y,pix_x,num_imgs)) # Pre-allocating the number of images

    for i in tqdm(range(0,num_imgs)):
        spad_com_img[:,:,i] = calc_img(pix_y,pix_x,peak_posi,sig)#calculating each image

    
    return spad_com_img


# def monte_carlo(true_depth, absorption_coefficient, scattering_coefficient, num_photons):
#     height, width = true_depth.shape
#     depth_max = true_depth.max() # 最大深度（单位：像素）

#     # 初始化图像
#     image = np.zeros((height, width))

#     # 定义散射和吸收参数
#     absorption_coefficient
#     scattering_coefficient

#     # 执行光线传播模拟
#     for i in range(num_photons):
#         x = np.random.randint(width)
#         y = np.random.randint(height)
#         depth = 0
        
#         while 0 <= x < width and 0 <= y < height and depth < depth_max:
#             # 模拟光线传播
#             depth += 0.15
#             x += np.random.choice([-1, 0, 1])
#             y += np.random.choice([-1, 0, 1])
            
#             # 模拟光子被吸收的概率
#             if np.random.rand() < absorption_coefficient:
#                 break
            
#             # 更新图像中的光强度
#             if 0 <= x < width and 0 <= y < height:
#                 image[y, x] += scattering_coefficient

            
#     return true_depth-image

def monte_carlo(true_depth, back_scattering_coefficient, step_sample_num):
    height, width = true_depth.shape
    depth_max = np.amax(true_depth) # 最大深度（单位：像素）
    depth_min = np.amin(true_depth)
    # 初始化图像
    image = np.zeros((height, width))

    # 定义散射和吸收参数
    back_scattering_coefficient
    step_num = int(height*width*back_scattering_coefficient)
    for step_depth in np.linspace(0, depth_min, step_num):
        for _ in range(step_sample_num):
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            x_idx = int(np.clip(x, 0, width - 1))
            y_idx = int(np.clip(y, 0, height - 1))
            true_depth[y_idx,x_idx]= step_depth
    return true_depth


if __name__ == '__main__':
    #Reads in the surface normal image from Unreal
    normal = cv2.imread("data_zhoucheng/new/Surf_Norm.EXR",cv2.IMREAD_UNCHANGED)
    normal = cv2.cvtColor(normal[:,:,0:3],cv2.COLOR_BGR2RGB)

    base_c = cv2.imread("data_zhoucheng/new/Base_Col.EXR",cv2.IMREAD_UNCHANGED)

    cam_vec = np.asarray([1,0,0]) # 相机x轴朝前
    cam_vec_mag = np.sqrt(np.sum(cam_vec**2)) # cam_vac的数值大小=1
    norm_mag = np.sqrt(np.sum(normal**2,axis=2))#归一化的cam_vec数值大小
    norm_prod = cam_vec_mag*norm_mag # 归一化乘积
    dot_p1 = np.sum(cam_vec*normal,axis=2) # 点乘后的第一个分量

    #The lambertian scattering angle defined with respect to the surface normal. Divison by 0 exceptions are handeled by the where flag.
    lambertian_ang = 1-((np.divide(dot_p1, norm_prod, out=np.zeros_like(dot_p1), where=norm_prod!=0)+1)/2) # 朗伯辐射角

    # read RGBD
    RGBD = cv2.imread("data_zhoucheng/new/RGBD.EXR",cv2.IMREAD_UNCHANGED)

    inten = cv2.cvtColor(RGBD[:,:,0:3],cv2.COLOR_BGR2GRAY) # 颜色通道专为灰度
    depth = RGBD[:,:,3] # 第四个通道为深度真值

    c_speed = 2.99e8 # 光速
    # depth = cv2.medianBlur(depth,3) # 使用中值滤波消除UE造成的异常干扰因素
    depth = depth/100.0 # 单位从厘米转为米
    cv2.imwrite('data_zhoucheng/new/GT.png', depth)
    # depth[depth>depth.min()+0.5]=depth.min()+1
    depth[depth>18]=18

    # plt.imshow(depth,vmin = np.amin(depth),vmax = np.amin(depth)+5,interpolation='none')
    plt.imshow(depth,vmin = 13,vmax =18,interpolation='none')
    plt.colorbar()
    plt.savefig('data_zhoucheng/new/depth.png')
    plt.close()
    # depth = monte_carlo(depth,0.03, 100)
    peak_pos = (depth/c_speed) # 深度真值转为时间
    peak_pos = peak_pos - np.amin(peak_pos) # 绝对时间转为相对时间

    #Initial image size, the y axis has been cropped from 192 to 176 pixels and the x axis has been double to 256 then cropped to 246
    img_x = 246
    img_y = 176

    peak_pos_2 = peak_pos

    c1 = np.arange(2,img_x,4) # 间隔4 取一列
    c2 = c1+1 # 所取列的右边相邻的一列

    c3 = np.vstack((c1,c2)).reshape((-1,),order='F') # 将其转为列的索引


    peak_pos_2 = peak_pos[2:174,c3]#将行减少为最终的大小，并且移除一半的列以达到2:1的sensor尺寸（172，122）
    # peak_pos_2 = peak_pos[c3,c3]#将行减少为最终的大小，并且移除一半的列以达到2:1的sensor尺寸（172，122）
    # # 加入后向散射
    # depth_truth_max = np.max(peak_pos_2)
    # depth_truth_min = np.min(peak_pos_2)
    

    peak_pos_2 = (peak_pos_2-np.amin(peak_pos_2)) # 转为相对时间

    peak_pos_2 = (peak_pos_2*2.0)+3.0e-9 # 时间乘2作为往返总时间，并加上一个最小深度时间作为offset



    # main
    bins = 10000 # 时间条个数

    num_frames = 10000 # 直方图所需的帧数

    num_imgs = 1 # 生成图的个数

    pix_x = 122 # 最终列数
    pix_y = 172 # 最终行数



    total_pix = pix_x*pix_y

    #histo_frames = np.zeros((pix_y,pix_x,target_frame_num))#Pre allocating click array

    jitter = 220e-12 # inter-pulse jitter

    f_num = 2.0 # 光圈数

    pixel_jit_map_max = 1.0e-9 # maximum inter-frame inter-pixel column wise jitter
    pixel_jit_map_min = pixel_jit_map_max/4.0 # minimum inter-frame inter-pixel column wise jitter

    pixel_jit_sig = np.linspace(pixel_jit_map_min/6.0,pixel_jit_map_max/6.0,pix_x) # An  inter-frame inter-pixel jitter vector varying linearly across the SPAD

    pixel_jit_map = np.zeros((pix_y,pix_x)) # Pre allocating the inter-frame inter-pixel jitter array 预分配间帧间像素抖动数组

    C_ref = 0.08956 # 目标物体反射率
    bckg_refs = 0.08956
    # C_ref = 0.88 # 目标物体反射率
    # bckg_refs = 0
    final_img = make_imgs(bins,np.asarray(peak_pos_2+pixel_jit_map),jitter,C_ref,bckg_refs,num_frames,f_num,num_imgs,pix_x,pix_y) # Simulating the image

    # backscatter = monte_carlo(final_img[:,:,0], 0.002, 100)
    final_img[:,:,0][final_img[:,:,0]<0]=0
    final_img[:,:,0][final_img[:,:,0]>160]=160

    plt.imshow(final_img[:,:,0],vmin = 40,vmax = 200, interpolation='none', aspect=0.5)
    # plt.imshow(final_img[:,:,0],vmin =,vmax = 16,interpolation='none')
    # plt.imshow(final_img[:,:,0],vmin = 60,vmax = 200,interpolation='none', aspect='0.5')
    plt.colorbar()
    plt.savefig('data_zhoucheng/new/fisher.png')  
    plt.close()


    backscatter = monte_carlo(final_img[:,:,0], 0.01, 30)
    cv2.imwrite('data_zhoucheng/new/shidi_300.png', cv2.resize(backscatter,(244,172)))
    plt.imshow(500-backscatter,cmap='Spectral',vmin = 0,vmax = 500,interpolation='none', aspect=0.5)
    # plt.imshow(final_img[:,:,0],vmin =,vmax = 16,interpolation='none')
    # plt.imshow(final_img[:,:,0],vmin = 60,vmax = 200,interpolation='none', aspect='0.5')
    plt.colorbar()
    plt.savefig('data_zhoucheng/new/shidi_300_visualize.png')  
