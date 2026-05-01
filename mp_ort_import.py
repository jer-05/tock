import torch.multiprocessing as mp
def exec_main(main):
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ONNX_WEIGHTS_DIR = "weights/onnx"
    import os
    import multiprocessing
    os.system(f"rm -r {ONNX_WEIGHTS_DIR}")
    os.system(f"mkdir {ONNX_WEIGHTS_DIR}")
    with multiprocessing.Manager() as manager:   
        currently_exporting_arr = manager.dict({})
        exported_models_cache = manager.dict({})
        export_lock = manager.Lock()
        ort_info = [currently_exporting_arr, export_lock, exported_models_cache]
        main(ort_info)
