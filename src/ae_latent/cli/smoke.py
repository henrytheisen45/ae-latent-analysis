def main():
    import ae_latent
    from ae_latent.models import unet_ae
    from ae_latent.data import dataset_utils
    from ae_latent.training import train_utils

    print("ae_latent:", ae_latent.__file__)
    print("model:", unet_ae.__file__)
    print("data:", dataset_utils.__file__)
    print("training:", train_utils.__file__)
    print("smoke_all ok")

if __name__ == "__main__":
    main()
