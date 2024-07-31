import numpy as np
from tqdm import tqdm
import torch
import SimpleITK as sitk
import torch.nn.functional as F

material = ['Air', 'Adipose', 'Muscle']

def load_itkfilewithtrucation(filename, upper=200, lower=-200, vmax=255, vmin=0):
    """
    load mhd files,set truncted value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """
    # 1,tructed outside of liver value
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    # 2,get tructed outside of liver value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    sitktructedimage.SetDirection(srcitkimage.GetDirection())
    # 3 normalization value to 0-255
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(vmax)
    rescalFilt.SetOutputMinimum(vmin)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return itkimage


def DecomposeTo3Materials(img_path, linear_cofficient, net, z_range):
    img_file = load_itkfilewithtrucation(img_path, upper=3024, lower=-3024, vmax=1.0, vmin=-1.0)
    img_array = sitk.GetArrayFromImage(img_file)
    print(img_array.shape)
    slice_list = []

    for z in tqdm(range(z_range[0], z_range[1]+1)):
        img = img_array[z]
        img = img.astype(np.float32)
        img = torch.tensor(img).cuda()
        img = torch.unsqueeze(torch.unsqueeze(img, 0), 0)
        net.eval()
        with torch.no_grad():
            output_img = net(img)
            output_img = torch.clip(output_img, min=-1., max=1.)
        denoise_img_show = output_img[0][0].detach().cpu().numpy()
        output_img = output_img[0][1].detach().cpu().numpy()

        linear_coefficient_high = output_img.reshape(-1)
        linear_coefficient_low = denoise_img_show.reshape(-1)
        
        Adipose_volume = np.zeros_like(linear_coefficient_high)
        Air_volume = np.zeros_like(linear_coefficient_high)
        Muscle_volume = np.zeros_like(linear_coefficient_high)
        material_list = [linear_cofficient[j] for j in material]
        for k in range(linear_coefficient_high.size):
            Micro = np.array([[linear_coefficient_low[k]], [linear_coefficient_high[k]], [1]])
            count = 0

            M = np.array([[material_list[0][0], material_list[1][0], material_list[2][0]],
                        [material_list[0][1], material_list[1][1], material_list[2][1]], [1, 1, 1]])
            alpha = np.dot(np.linalg.pinv(M), Micro)
            if np.sum((alpha >= 0) & (alpha <= 1)) == 3:
                eval(material[0] + '_volume')[k] = alpha[0]
                eval(material[1] + '_volume')[k] = alpha[1]
                eval(material[2] + '_volume')[k] = alpha[2]
            else:
                alpha = F.softmax(torch.from_numpy(alpha), dim=0).numpy()
                eval(material[0] + '_volume')[k] = alpha[0]
                eval(material[1] + '_volume')[k] = alpha[1]
                eval(material[2] + '_volume')[k] = alpha[2]

        Air_volume = Air_volume.reshape(output_img.shape)
        Adipose_volume = Adipose_volume.reshape(output_img.shape)
        Muscle_volume = Muscle_volume.reshape(output_img.shape)

        material_data = np.stack((Adipose_volume, Air_volume, Muscle_volume), axis=0)
        slice_list.append(material_data)

    final_result = np.stack(slice_list, axis=1)
    return final_result
