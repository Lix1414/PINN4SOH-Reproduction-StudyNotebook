"""验证数据、模型、损失、训练与评估的端到端流程。"""

from module_loader import load_clean_module


dataloader_module = load_clean_module("02_dataloader.py", "verify_data")
model_module = load_clean_module("03_model.py", "verify_model")
train_module = load_clean_module("05_train.py", "verify_train")
eval_module = load_clean_module("06_eval.py", "verify_eval")


def main(epochs=3):
    """执行最小端到端验证并返回结果。"""
    train_files, test_files = dataloader_module.split_xjtu_files()
    train_bundle = dataloader_module.build_dataloaders(train_files[:2], 2.0, 128)
    test_bundle = dataloader_module.build_dataloaders(test_files[:1], 2.0, 128)
    model = model_module.PINN()
    history = train_module.Train(
        model, train_bundle["train_2"], train_bundle["valid_2"],
        test_bundle["test_3"], epochs=epochs,
        warmup_epochs=min(2, epochs - 1), warmup_lr=0.001,
        base_lr=0.005, final_lr=0.0005,
        early_stop_patience=None,
    )
    true_label, pred_label = model.Test(test_bundle["test_3"])
    mae, mape, mse, rmse = eval_module.eval_metrix(true_label, pred_label)
    result = {
        "train_pairs": len(train_bundle["train_2"].dataset),
        "valid_pairs": len(train_bundle["valid_2"].dataset),
        "test_pairs": len(test_bundle["test_3"].dataset),
        "epochs": len(history["epoch"]),
        "MAE": mae, "MAPE_percent": mape, "MSE": mse, "RMSE": rmse,
    }
    for key, value in result.items():
        print(f"{key}: {value}")
    return result


if __name__ == "__main__":
    main()
