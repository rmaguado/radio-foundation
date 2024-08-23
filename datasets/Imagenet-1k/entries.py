from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(
        split=split,
        root="./datasets/imagenet/ILSVRC2012/",
        extra="./datasets/imagenet/extra",
    )
    dataset.dump_extra()
