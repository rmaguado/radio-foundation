from dinov2.data.datasets import LidcIdri

for split in LidcIdri.Split:
    dataset = LidcIdri(split=split, root="./datasets/lidc/LIDC_IDRI", extra="./datasets/lidc/extra")
    dataset.dump_extra()