"""加载PINN模型与XJTU演示数据。"""

from module_loader import load_clean_module


model_module = load_clean_module("03_model.py", "clean_model")
dataloader_module = load_clean_module("02_dataloader.py", "clean_dataloader")
PINN = model_module.PINN


def get_demo_dataloader():
    """返回XJTU 2C训练DataLoader。"""
    train_files, _ = dataloader_module.split_xjtu_files()
    loaders = dataloader_module.build_dataloaders(train_files, 2.0)
    return loaders["train_2"]


def get_model_and_data():
    """返回未训练PINN和演示DataLoader。"""
    return PINN(), get_demo_dataloader()


if __name__ == "__main__":
    model, loader = get_model_and_data()
    print(type(model).__name__)
    print(len(loader.dataset))
