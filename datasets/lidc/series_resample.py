import SimpleITK as sitk

def resample_series_to_nii(folderpath, destination):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folderpath)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    current_spacing = image.GetSpacing()
    current_size = image.GetSize()

    desired_spacing = (current_spacing[0], current_spacing[0], current_spacing[0])
    desired_size = [int(round(current_size[i] * (current_spacing[i] / desired_spacing[i]))) for i in range(3)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(desired_spacing)
    resample.SetSize(desired_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resampled_image = resample.Execute(image)
    
    sitk.WriteImage(resampled_image, destination) #.nii