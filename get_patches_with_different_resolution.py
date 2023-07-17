### get patches with different resolutions


dataset = sorted(glob.glob("training/counts/*/*/*.npz"))  # the npz collection for each count file

patients = sorted(get_spatial_patients().keys())

test_patients = ["BC23450", "BC23903"]
train_patients = [p for p in patients if p not in test_patients]

print("Train patients: ",  train_patients)
print("Test patients: ", test_patients)
print()  

print(dataset)


dataset = [d for d in dataset if ((d.split("/")[-2] in train_patients) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in train_patients))]
    

for re in [128, 256, 299, 384, 512]:

    print("Saving and cropping patches with resolution : " + str(re))

    train_dataset = SubGenerator(train_patients, window = 299, resolution =re,
                            img_cached = 'training/images/',
                            transform=torchvision.transforms.ToTensor()) # range [0, 255] -> [0.0,1.0]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, 
                            num_workers=0, shuffle=True)

    for (i, (he, npz)) in enumerate(train_loader):
            # calculate the white ratio and delete the noise images
        print("Saving training filtered images and counts: " + str((i + 1) * 32) + "/" + str(len(train_dataset)) + '...')


    test_dataset = SubGenerator(test_patients, window = 299, resolution =re,
                            img_cached = 'test/images/',
                            transform=torchvision.transforms.ToTensor()) # range [0, 255] -> [0.0,1.0]

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, 
                            num_workers=0, shuffle=True)

    print(len(test_dataset))

    for (i, (he, npz)) in enumerate(test_loader): 
        print("Saving test filtered images and counts: " + str((i + 1) * 32) + "/" + str(len(test_dataset)) + '...')
